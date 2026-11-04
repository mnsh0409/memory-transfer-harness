"""S2 — Graphiti adapter (temporal knowledge graph with bi-temporal edge invalidation).

Upstream: `graphiti-core` (getzep/graphiti), version pinned in requirements.lock.
Import-guarded like the other adapters.

Graphiti is the heaviest system in the grid and the one whose plumbing diverges most from
what the protocol assumed. What this adapter does, and why:

* **No Docker, no Neo4j — the graph is embedded Kuzu.** The protocol assumes Neo4j via
  docker-compose; the evaluation machine had no Docker daemon. graphiti-core ships a `KuzuDriver` that
  runs in-process, including `db=":memory:"`. That is what this adapter uses by default. Kuzu
  is marked deprecated upstream (its own maintenance stalled), so this is a *portability*
  choice made under deadline, not a preference — pass `uri=`/`user=`/`password=` to run
  against a real Neo4j instead, and the rest of the adapter is unchanged.
* **Per-dialogue isolation is a fresh graph per dialogue, not just `group_id`.** The plan
  says `group_id = dialogue_id`, and this adapter does set it — but graphiti-core 0.29.3 has
  **no `delete_group`** (the API in the docs belongs to a different version), so `reset()`
  could not actually drop a dialogue's data. A new in-memory graph per dialogue makes
  isolation structural rather than a filter that must be remembered on every call.
* **Episodes are ingested strictly sequentially.** Graphiti resolves each episode against the
  graph built so far; concurrent ingestion of one dialogue's turns would make the resulting
  graph order-dependent and irreproducible. The `MemorySystem` contract is sequential
  already, and this adapter never batches.
* **`edge_invalidation` flag (the frozen scope ablation 3).** Upstream exposes no switch. The seam
  is `resolve_edge_contradictions`, which already returns `[]` when a new edge contradicts
  nothing; with the flag off we patch it to always return `[]`, so no edge is ever expired.
  That is the library's own no-contradiction path, not a reimplementation. The patch is
  applied to the module attribute (upstream resolves the name at call time) and is therefore
  **process-global** — one condition per process, which is how the grid runs.
* **Structured output.** Graphiti requires JSON conforming to a pydantic model per call, and
  the harness backend takes a prompt and returns text. `HarnessLLMClient` appends the
  requested JSON schema to the prompt and parses the reply. This is a real deviation from
  "as shipped" (the shipped client uses the provider's structured-output mode) and is the
  one place where the adapter changes what Graphiti sends. It is unavoidable without letting
  Graphiti open its own uncached, non-temp-0 connection (ARCHITECTURE.md).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from harness.memory_base import MemoryEntry, MemorySystem, Turn

try:
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.llm_client.client import LLMClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.nodes import EpisodeType
    HAVE_GRAPHITI = True
except Exception:
    HAVE_GRAPHITI = False
    LLMClient = object  # type: ignore
    CrossEncoderClient = object  # type: ignore
    EmbedderClient = object  # type: ignore

SOURCE_DESCRIPTION = "conversation turn (replay)"


class RefusingCrossEncoder(CrossEncoderClient):  # type: ignore[misc]
    """Default reranker for this adapter: refuses instead of calling OpenAI.

    Graphiti builds an `OpenAIRerankerClient` when none is supplied — an LLM call that would
    bypass `harness.llm`, the cache, and the temp-0 rule, and that no manifest would record
    (ARCHITECTURE.md). Measured on the real library: the search path this adapter uses never invokes
    the reranker (`cross_encoder used: 0` across ingest, search and export). So refusing costs
    nothing today and converts a future silent-spend regression into a loud failure. Pass a
    real cross-encoder explicitly if a recipe ever needs one."""

    async def rank(self, query: str, passages: list[str]):   # noqa: ARG002
        raise RuntimeError(
            "Graphiti asked for a cross-encoder rerank. This adapter deliberately does not "
            "provide one: the shipped default is an OpenAI reranker whose calls would bypass "
            "harness.llm, the cache and temp 0. Pass cross_encoder=... explicitly and record "
            "why before enabling it."
        )


class SentenceTransformerEmbedder(EmbedderClient):  # type: ignore[misc]
    """Local embedder (sentence-transformers) for offline smoke tests.

    NOT the default: Graphiti ships with an OpenAI embedder and we evaluate systems as
    shipped, so real runs use that. This exists so the adapter's live test can exercise the
    real ingest/search path with no API key."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model_name, local_files_only=True)

    async def create(self, input_data):
        if isinstance(input_data, str):
            return self._m.encode(input_data).tolist()
        items = list(input_data)
        return self._m.encode(items[0] if items else "").tolist()

    async def create_batch(self, input_data_list):
        return [v.tolist() for v in self._m.encode(list(input_data_list))]


class HarnessLLMClient(LLMClient):  # type: ignore[misc]
    """graphiti_core `LLMClient` backed by `harness.llm.LLMBackend` (temp 0, cached).

    Graphiti calls `_generate_response(messages, response_model, ...)` and expects a dict
    matching `response_model`. The schema is appended to the prompt AND the provider's
    structured-output mode is requested, because §4.1 of the paper states that every library
    which parses JSON out of a reply is given that mode — Graphiti was the one adapter still
    relying on schema-in-prompt alone, which made that sentence false as written and produced
    observably malformed replies (`{"edges":[]}}`, salvaged only by `json_repair`).

    A reply that cannot be parsed raises, because returning `{}` would let Graphiti silently
    record an episode with no extracted entities — a quietly empty graph is far worse than a
    loud failure."""

    def __init__(self, backend, config: "LLMConfig | None" = None) -> None:
        super().__init__(config or LLMConfig(api_key="unused-harness-backend", temperature=0.0))
        self._backend = backend
        self.calls = 0

    @staticmethod
    def _flatten(messages) -> str:
        return "\n\n".join(f"{getattr(m, 'role', 'user')}: {getattr(m, 'content', '')}"
                           for m in messages)

    async def _generate_response(self, messages, response_model=None,
                                 max_tokens=None, model_size=None, **kwargs) -> dict:  # noqa: ARG002
        prompt = self._flatten(messages)
        response_format = None
        if response_model is not None:
            schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
            prompt += ("\n\nRespond with JSON only, conforming exactly to this JSON schema. "
                       "Output the JSON object and nothing else.\n" + schema)
            # The schema stays in the prompt as well: json_object mode guarantees *valid* JSON,
            # not JSON of this shape, and Graphiti's parser needs the shape.
            response_format = {"type": "json_object"}
        self.calls += 1
        text = self._backend.complete(prompt, response_format=response_format)
        parsed = self._parse(text)
        if parsed is None:
            raise ValueError(
                "Graphiti asked for structured output and the backend reply did not parse as "
                f"JSON (first 200 chars: {text[:200]!r}). Refusing to return an empty result: "
                "that would silently produce a graph with no entities."
            )
        return parsed

    @staticmethod
    def _parse(text: str):
        text = (text or "").strip()
        if text.startswith("```"):                       # strip a ```json fence
            text = text.split("```")[1] if "```" in text[3:] else text[3:]
            text = text[4:] if text.lower().startswith("json") else text
        try:
            out = json.loads(text)
            return out if isinstance(out, dict) else None
        except Exception:
            pass
        try:
            from json_repair import repair_json
            out = json.loads(repair_json(text))
            return out if isinstance(out, dict) else None
        except Exception:
            return None


class GraphitiMemory(MemorySystem):
    name = "S2_graphiti"
    capabilities = frozenset({"edge_invalidation"})

    def __init__(self, backend=None, edge_invalidation: bool = True, embedder=None,
                 cross_encoder=None, db: str = ":memory:", uri: str = None,
                 user: str = None, password: str = None) -> None:
        if not HAVE_GRAPHITI:
            raise RuntimeError(
                "graphiti-core not installed. pip install graphiti-core kuzu "
                "(version in requirements.lock)."
            )
        if backend is None:
            raise ValueError(
                "Graphiti calls an LLM on every episode; pass the harness LLMBackend so calls "
                "are cached and temp-0 (ARCHITECTURE.md). It has no offline mode."
            )
        self._backend = backend
        self._edge_invalidation = edge_invalidation
        self._embedder = embedder
        self._cross_encoder = cross_encoder
        self._db = db
        self._neo4j = (uri, user, password) if uri else None
        self._dialogue_id: str | None = None
        self._g = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._episode_n = 0
        if not edge_invalidation:
            _disable_edge_invalidation()

    # ------------------------------------------------------------------
    def _run(self, coro):
        """Graphiti is async; MemorySystem is sync. One private loop for the adapter's
        lifetime — `asyncio.run` per call would tear down the driver's connections."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def _graphiti_for(self, dialogue_id: str):
        if self._g is not None and self._dialogue_id == dialogue_id:
            return self._g
        self._close()
        kwargs = {"llm_client": HarnessLLMClient(self._backend),
                  "cross_encoder": self._cross_encoder or RefusingCrossEncoder()}
        if self._embedder is not None:
            kwargs["embedder"] = self._embedder
        if self._neo4j:
            uri, user, password = self._neo4j
            self._g = Graphiti(uri=uri, user=user, password=password, **kwargs)
        else:
            driver = KuzuDriver(db=self._db)
            # UPSTREAM BUG WORKAROUND (graphiti-core 0.29.3): add_episode does
            # `if group_id != self.driver._database: self.driver = self.driver.clone(...)`,
            # treating group_id as a database name for multi-tenancy. GraphDriver only
            # *annotates* `_database`; KuzuDriver never assigns it, so passing any group_id
            # raises AttributeError. Kuzu is a single embedded database and we already give
            # each dialogue its own graph, so declaring this graph's database to BE the
            # dialogue is both true and exactly what stops the pointless clone.
            driver._database = dialogue_id
            self._g = Graphiti(graph_driver=driver, **kwargs)
        self._run(self._g.build_indices_and_constraints())
        self._run(_ensure_kuzu_fulltext_indices(self._g))
        self._dialogue_id = dialogue_id
        self._episode_n = 0
        return self._g

    def _close(self) -> None:
        if self._g is not None:
            try:
                self._run(self._g.close())
            except Exception:
                pass
        self._g = None
        self._dialogue_id = None

    # ------------------------------------------------------------------
    def ingest_turn(self, dialogue_id: str, turn: Turn) -> None:
        g = self._graphiti_for(dialogue_id)
        self._episode_n += 1
        # reference_time must advance monotonically within a dialogue or the bi-temporal
        # model cannot order episodes; replay has no wall-clock timestamps, so turn order is
        # the only honest clock. Derived from the episode counter, not datetime.now(), so a
        # re-run reproduces the same graph.
        reference_time = datetime(2026, 1, 1, tzinfo=timezone.utc).replace(
            minute=self._episode_n % 60, hour=(self._episode_n // 60) % 24)
        self._run(g.add_episode(
            name=turn.turn_id,
            episode_body=f"{turn.role}: {turn.text}",
            source_description=SOURCE_DESCRIPTION,
            reference_time=reference_time,
            source=EpisodeType.message,
            group_id=dialogue_id,          # the isolation key; the graph is per-dialogue too
        ))

    def query(self, dialogue_id: str, utterance: str, k: int) -> list[MemoryEntry]:
        g = self._graphiti_for(dialogue_id)
        edges = self._run(g.search(query=utterance, group_ids=[dialogue_id], num_results=k))
        return [
            MemoryEntry(
                entry_id=str(getattr(e, "uuid", i)),
                text=getattr(e, "fact", "") or "",
                meta={"kind": "graphiti_edge",
                      "valid_at": str(getattr(e, "valid_at", "") or ""),
                      "invalid_at": str(getattr(e, "invalid_at", "") or ""),
                      "expired_at": str(getattr(e, "expired_at", "") or "")},
            )
            for i, e in enumerate(edges or [])
        ][:k]

    def reset(self, dialogue_id: str) -> None:
        """Drop the whole graph for this dialogue. graphiti-core 0.29.3 has no delete_group,
        so a fresh graph is the only reliable reset (module docstring)."""
        self._close()

    def export_store(self, dialogue_id: str) -> list[MemoryEntry]:
        """TRUE DUMP: every entity edge (fact) in this dialogue's graph, read with a Cypher
        query rather than `search()` — a search would rank and cap, which is not a dump.
        Expired/invalidated edges are included and flagged, because whether a system correctly
        expired a fact is exactly what the temporal ablation is about."""
        g = self._graphiti_for(dialogue_id)
        # Kuzu models RELATES_TO as an intermediate NODE table (`RelatesToNode_`) rather than
        # a relationship with properties, so the Neo4j-shaped `MATCH ()-[e:RELATES_TO]->()`
        # fails with "Cannot find property group_id for e". Pick the query per provider.
        if getattr(g.driver, "provider", None) is not None and g.driver.provider.value == "kuzu":
            cypher = ("MATCH (e:RelatesToNode_) WHERE e.group_id = $group_id "
                      "RETURN e.uuid AS uuid, e.fact AS fact, e.valid_at AS valid_at, "
                      "e.invalid_at AS invalid_at, e.expired_at AS expired_at")
        else:
            cypher = ("MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $group_id "
                      "RETURN e.uuid AS uuid, e.fact AS fact, e.valid_at AS valid_at, "
                      "e.invalid_at AS invalid_at, e.expired_at AS expired_at")
        records, _, _ = self._run(g.driver.execute_query(cypher, group_id=dialogue_id))
        out = []
        for i, r in enumerate(records or []):
            fact = r.get("fact") or ""
            if not fact:
                continue
            out.append(MemoryEntry(
                entry_id=str(r.get("uuid", i)),
                text=fact,
                meta={"kind": "graphiti_edge",
                      "valid_at": str(r.get("valid_at") or ""),
                      "invalid_at": str(r.get("invalid_at") or ""),
                      "expired_at": str(r.get("expired_at") or ""),
                      "expired": bool(r.get("expired_at"))},
            ))
        return out


async def _ensure_kuzu_fulltext_indices(g) -> None:
    """SECOND UPSTREAM BUG WORKAROUND (graphiti-core 0.29.3).

    `KuzuDriver.build_indices_and_constraints` is a **no-op** whose comment claims indices are
    created by `setup_schema()` — they are not: `setup_schema` runs the static schema only,
    and the full-text indices live in `get_fulltext_indices(KUZU)`, which is called from
    `KuzuGraphMaintenanceOperations.build_indices_and_constraints` — a method nothing invokes.
    The result is that any search reaching the edge full-text index dies with
    "Table RelatesToNode_ doesn't have an index with name edge_name_and_fact", and it dies
    during *ingest* (edge dedup searches), not just at query time.

    So call the driver's own operations object directly. Harmless on non-Kuzu drivers (skipped)
    and idempotent-ish: re-creating an existing FTS index raises, which is swallowed."""
    driver = getattr(g, "driver", None)
    ops = getattr(driver, "graph_ops", None)
    if driver is None or ops is None or driver.provider.value != "kuzu":
        return
    try:
        await ops.build_indices_and_constraints(driver)
    except Exception:
        pass    # already present (a rebuilt graph reuses an existing on-disk database)


def _disable_edge_invalidation() -> None:
    """the frozen scope ablation 3. Patches the module attribute upstream looks up at call time, so
    no contradicted edge is ever expired. Process-global by necessity — see module docstring."""
    from graphiti_core.utils.maintenance import edge_operations

    if getattr(edge_operations, "_replay_invalidation_disabled", False):
        return
    edge_operations._replay_original_resolve_edge_contradictions = (
        edge_operations.resolve_edge_contradictions
    )
    edge_operations.resolve_edge_contradictions = lambda resolved_edge, candidates: []
    edge_operations._replay_invalidation_disabled = True


def _restore_edge_invalidation() -> None:
    """Undo `_disable_edge_invalidation` (tests; and any process that runs both arms)."""
    from graphiti_core.utils.maintenance import edge_operations

    original = getattr(edge_operations, "_replay_original_resolve_edge_contradictions", None)
    if original is not None:
        edge_operations.resolve_edge_contradictions = original
        edge_operations._replay_invalidation_disabled = False

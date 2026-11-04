# Harness architecture

Design decisions behind the replay harness.

1. EVERYTHING IS A MemorySystem. Baselines B0 (empty), B1 (verbatim history),
   B2 (gold PTKB) are degenerate implementations, so replay → rewrite →
   retrieve → rerank is ONE code path and conditions differ only in the
   `condition` field of RunConfig.
2. REPLAY ORDER IS THE PROTOCOL. query() strictly before ingest() of the
   current turn; per-dialogue isolation; temp 0 everywhere.
3. EVERY LLM CALL IS CACHED. Content-hash key (backend, model, temp, prompt) →
   JSONL. Re-runs free; provider drift detectable; cache is a releasable
   artifact (contains public topic/PTKB text only, never licensed passages).
4. RUN IDENTITY = CONFIG HASH. RunConfig canonical JSON → run_id → append-only
   artifacts. No overwrites, ever.
5. FORMAT-NEUTRAL MEMORY BLOCK. Adapters return MemoryEntry(text); replay
   renders one-per-line verbatim. No system's native structure gets a prompt
   advantage.
6. ADAPTER CONTRACT: ingest_turn / query / reset / export_store (+capabilities
   set for ablation flags: infer, evolution, edge_invalidation, sleeptime).
   export_store feeds intrinsic eval — must be a true dump, not a search.
7. STAGES DECOUPLED. Rewrites are computed once per (condition, collection);
   retrieval and reranking consume stored rewrites, so both injection modes
   reuse the same replay pass.

## Invariants

The rules every change to the harness must keep. Comments across `src/` and `tests/` refer
back to these, and the tests enforce most of them.

- **One code path.** Baselines are degenerate `MemorySystem`s (`src/harness/memory_base.py`).
  If a change special-cases a baseline downstream, the design is wrong.
- **Query before ingest.** `replay.py` calls `memory.query()` before ingesting the current
  turn. Tests enforce the order; do not reorder.
- **Per-dialogue isolation.** Every system gets a separate store per dialogue
  (`user_id`, `group_id`, a fresh graph, or a separate directory, as the library allows).
- **Every model call goes through the harness.** All calls run at temperature 0 through
  `src/harness/llm.py` and the content-hashed cache in `src/harness/llmcache.py`; a raw API
  call anywhere else in experiment code is a bug. Where a library opens its own connection
  that cannot be routed (Mem0's extractor, Graphiti's embedder), the adapter says so and the
  paper reports that system's cost as a lower bound.
- **Pinned versions, never upgraded silently.** `mem0ai` stays pre-v3 (v3 removed the
  extract → ADD/UPDATE/DELETE pipeline the published benchmarks measured); A-Mem and GAM are
  pinned by commit. Exact versions are in `requirements.lock`. The model is one pinned snapshot
  (`gpt-4o-mini-2024-07-18`), never an alias.
- **Frozen instruments.** `prompts/rewrite_v1.txt` and `prompts/judge_v1.txt` are never edited.
  A new instrument is a new versioned file.
- **Append-only runs.** Every run writes a manifest, and its identity is the hash of its
  config (`src/harness/config.py`). A TREC run file is never overwritten; a new config is a new
  run ID.
- **No licensed collection text.** No ClueWeb22 or iKAT passage text goes into code, commits,
  logs, or model prompts beyond what retrieval requires. The repository and the LLM cache hold
  public topic and PTKB text only.

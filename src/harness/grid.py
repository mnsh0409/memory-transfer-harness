"""Grid definition — the exact set of runs this paper reports.

Every entry here traces to the experiment's frozen scope, fixed before any system was run. To
change what runs, change the scope deliberately and then this file to match; do not add a
config here that the scope doesn't sanction.

Ordering matters and is not cosmetic: `main_grid()` comes before `ablation_grid()` because
the frozen scope says ablations queue only after the main grid is complete. `emit_grid` writes an
explicit numeric prefix into each filename so a shell `for cfg in runs/configs/*.json` loop
runs them in the right order by default.
"""
from __future__ import annotations

import json
import os

from .config import RunConfig

# the frozen scope "Frozen: conditions" -- B2_gold_ptkb is deliberately absent (oracle is out for
# this paper; it survives in the codebase only to keep the one-code-path design intact).
BASELINES = ("B0_none", "B1_full_history")
SYSTEMS = ("S1_mem0", "S2_graphiti", "S3_amem", "S4_letta", "S5_gam")

# the frozen scope "Frozen: data"
COLLECTIONS = ("cast19", "cast20")

# the frozen scope "Frozen: injection mode" -- rewrite_only for the entire main grid.
INJECTION = "rewrite_only"

GATE_CONDITIONS = ("B0_none", "B1_full_history")
GATE_COLLECTION = "cast19"


def gate_grid() -> list[RunConfig]:
    """The B1>B0 gate (the frozen scope "Frozen: the gate"). No S-run starts until this passes."""
    return [
        RunConfig(condition=c, injection=INJECTION, collection=GATE_COLLECTION)
        for c in GATE_CONDITIONS
    ]


def main_grid(context: str = "history_plus_memory") -> list[RunConfig]:
    """Baselines + all five systems, both collections, rewrite_only. 2 x 7 = 14 runs.

    Includes the two gate configs: their RunConfigs are identical to gate_grid()'s, so they
    resolve to the same run_id and the same already-written artifacts. Re-running them is a
    no-op that fails fast on the append-only guard rather than recomputing -- that's intended,
    and it's why the gate isn't a separate special-cased code path."""
    return [
        RunConfig(condition=c, injection=INJECTION, collection=coll, context=context)
        for coll in COLLECTIONS
        for c in (*BASELINES, *SYSTEMS)
    ]


def ablation_grid(best_system: str | None = None) -> list[RunConfig]:
    """the frozen scope "Frozen: ablations", in its stated priority order so that a truncated list is
    still a coherent story.

    best_system: required only for the k-sweep (ablation 4), which the frozen scope scopes to the
    single best-performing system by nDCG@3 on CAsT 2019. Pass it once the main grid is scored;
    with None, the k-sweep is omitted and the first three ablations are still emitted."""
    configs: list[RunConfig] = []

    # 1. Mem0 infer=False -- both collections (the frozen scope names no year restriction here).
    for coll in COLLECTIONS:
        configs.append(RunConfig(condition="S1_mem0", injection=INJECTION, collection=coll,
                                 ablation={"infer": False}))
    # 2. A-Mem evolution off.
    for coll in COLLECTIONS:
        configs.append(RunConfig(condition="S3_amem", injection=INJECTION, collection=coll,
                                 ablation={"evolution": False}))
    # 3. Graphiti edge-invalidation off -- 2019 ONLY, per the frozen scope.
    configs.append(RunConfig(condition="S2_graphiti", injection=INJECTION, collection="cast19",
                             ablation={"edge_invalidation": False}))
    # 4. Retrieval depth k in {3,5,10} on the best system only.
    if best_system is not None:
        if best_system not in SYSTEMS:
            raise ValueError(f"best_system must be one of {SYSTEMS}, got {best_system!r}")
        for k in (3, 5, 10):
            configs.append(RunConfig(condition=best_system, injection=INJECTION,
                                     collection="cast19", memory_k=k))
    return configs


def _with_context(config: RunConfig, context: str) -> RunConfig:
    """ablation_grid() builds configs at the default context; re-stamp them for the design
    being emitted. Context is part of the run_id, so this yields distinct artifacts rather
    than colliding with the history_plus_memory ablations."""
    return RunConfig(condition=config.condition, injection=config.injection,
                     collection=config.collection, backend=config.backend,
                     model=config.model, memory_k=config.memory_k,
                     ablation=dict(config.ablation), rewrite_prompt=config.rewrite_prompt,
                     context=context)


def emit_grid(out_dir: str, best_system: str | None = None,
              context: str = "history_plus_memory") -> list[str]:
    """Write every config to out_dir as JSON. Filenames carry a numeric prefix so that the
    protocol's `for cfg in runs/configs/*.json` loop hits the main grid before the ablations,
    which the frozen scope requires. Returns the paths written, in order.

    Refuses to overwrite: a config file that already exists means the grid was emitted before,
    and silently rewriting it would let a changed grid definition masquerade as the original.
    Delete the directory deliberately if you mean to regenerate."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    if context == "history_plus_memory":
        groups = [("10-main", main_grid()), ("20-ablation", ablation_grid(best_system))]
    else:
        # The memory_only design is a separate experiment, not an extension of the main grid
        # (B1 is a real ceiling there and a no-op here -- paper, Section 3.4), so it gets its own
        # filename prefix and never lands in a `10-main-*` loop by accident.
        #
        # It gets the frozen scope's ablations too. It did not originally, which was a mistake: the
        # ablation list is scoped to whichever design is reportable, and once memory_only
        # became that design (Aug 23) the ablations had to be hand-written to run at all.
        # Hand-written configs are how a grid definition and what actually ran drift apart.
        groups = [(f"15-{context}", main_grid(context)),
                  (f"25-{context}-ablation",
                   [_with_context(c, context) for c in ablation_grid(best_system)])]
    for prefix, configs in groups:
        for i, config in enumerate(configs):
            name = f"{prefix}-{i:02d}-{config.run_id}.json"
            path = os.path.join(out_dir, name)
            if os.path.exists(path):
                raise FileExistsError(
                    f"{path} exists -- the grid was already emitted. Delete {out_dir} "
                    "deliberately if you intend to regenerate it."
                )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_config_dict(config), f, indent=2, sort_keys=True)
                f.write("\n")
            written.append(path)
    return written


def _config_dict(config: RunConfig) -> dict:
    """RunConfig fields as a plain dict, suitable for `RunConfig(**json.load(f))` round-trip.
    Deliberately omits nothing: the emitted file is the complete, reproducible spec."""
    return {
        "condition": config.condition,
        "injection": config.injection,
        "collection": config.collection,
        "backend": config.backend,
        "model": config.model,
        "memory_k": config.memory_k,
        "ablation": dict(sorted(config.ablation.items())),
        "rewrite_prompt": config.rewrite_prompt,
        "context": config.context,
    }

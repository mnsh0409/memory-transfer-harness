"""Grid tests. These are really tests that the code agrees with the frozen scope -- each one names
the frozen scope clause it enforces, so a scope change that edits one without the other fails
here rather than silently producing the wrong set of runs."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from harness.config import RunConfig
from harness.grid import ablation_grid, emit_grid, gate_grid, main_grid


def test_main_grid_is_two_collections_by_seven_conditions():
    grid = main_grid()
    assert len(grid) == 14
    assert {c.collection for c in grid} == {"cast19", "cast20"}
    assert len({c.run_id for c in grid}) == 14  # every config distinct


def test_oracle_condition_is_out_of_the_grid():
    """the frozen scope 'Frozen: conditions': B2_gold_ptkb is OUT for this paper -- it only means
    something on the licensed corpus. It stays in config.CONDITIONS (removing it would break
    the one-code-path design) but must not appear in any emitted run."""
    from harness.config import CONDITIONS
    assert "B2_gold_ptkb" in CONDITIONS          # still a legal condition...
    assert RunConfig("B2_gold_ptkb", "rewrite_only", "cast19")   # ...still constructible...
    for config in (*main_grid(), *ablation_grid(), *gate_grid()):  # ...but never scheduled
        assert config.condition != "B2_gold_ptkb"


def test_entire_grid_is_rewrite_only():
    """the frozen scope 'Frozen: injection mode': rewrite_plus_rerank is cut."""
    for config in (*main_grid(), *ablation_grid("S1_mem0"), *gate_grid()):
        assert config.injection == "rewrite_only"


def test_gate_grid_is_b0_and_b1_on_cast19():
    grid = gate_grid()
    assert len(grid) == 2
    assert {c.condition for c in grid} == {"B0_none", "B1_full_history"}
    assert {c.collection for c in grid} == {"cast19"}


def test_gate_configs_are_identical_to_their_main_grid_counterparts():
    """The gate is not a special code path -- its configs must hash to the same run_ids as the
    corresponding main-grid entries, so gate artifacts are reused rather than recomputed."""
    gate_ids = {c.run_id for c in gate_grid()}
    main_ids = {c.run_id for c in main_grid()}
    assert gate_ids <= main_ids


def test_ablations_match_freeze_list():
    """the frozen scope 'Frozen: ablations', including the 2019-only restriction on Graphiti."""
    grid = ablation_grid()
    by_condition = {}
    for c in grid:
        by_condition.setdefault(c.condition, []).append(c)

    assert {c.collection for c in by_condition["S1_mem0"]} == {"cast19", "cast20"}
    assert all(c.ablation == {"infer": False} for c in by_condition["S1_mem0"])

    assert {c.collection for c in by_condition["S3_amem"]} == {"cast19", "cast20"}
    assert all(c.ablation == {"evolution": False} for c in by_condition["S3_amem"])

    graphiti = by_condition["S2_graphiti"]
    assert len(graphiti) == 1
    assert graphiti[0].collection == "cast19"  # 2019 ONLY per the frozen scope
    assert graphiti[0].ablation == {"edge_invalidation": False}


def test_k_sweep_omitted_without_best_system():
    assert all(c.memory_k == 5 for c in ablation_grid())  # 5 is RunConfig's default


def test_k_sweep_scoped_to_one_system_on_cast19():
    """the frozen scope: k in {3,5,10} on the single best-performing system only."""
    grid = ablation_grid(best_system="S1_mem0")
    sweep = [c for c in grid if c.memory_k != 5 or (c.condition == "S1_mem0" and not c.ablation)]
    ks = sorted({c.memory_k for c in grid if not c.ablation})
    assert ks == [3, 5, 10]
    assert all(c.collection == "cast19" for c in grid if not c.ablation)
    assert {c.condition for c in grid if not c.ablation} == {"S1_mem0"}
    assert sweep  # sanity: the sweep is non-empty


def test_k_sweep_rejects_a_non_system():
    with pytest.raises(ValueError):
        ablation_grid(best_system="B0_none")


def test_emit_grid_orders_main_before_ablations(tmp_path):
    """the frozen scope requires ablations queue only after the main grid. The protocol drives runs
    with `for cfg in runs/configs/*.json`, which is alphabetical -- so the filename prefixes
    have to encode that order."""
    paths = emit_grid(str(tmp_path))
    names = [os.path.basename(p) for p in paths]
    assert names == sorted(names), "glob order must match intended run order"
    first_ablation = next(i for i, n in enumerate(names) if n.startswith("20-ablation"))
    assert all(n.startswith("10-main") for n in names[:first_ablation])


def test_emitted_configs_round_trip_into_runconfig(tmp_path):
    for path in emit_grid(str(tmp_path)):
        with open(path, encoding="utf-8") as f:
            config = RunConfig(**json.load(f))
        assert config.run_id in os.path.basename(path)


def test_emit_grid_refuses_to_overwrite(tmp_path):
    emit_grid(str(tmp_path))
    with pytest.raises(FileExistsError):
        emit_grid(str(tmp_path))

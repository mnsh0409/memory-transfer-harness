"""Tests for scripts/judge_agreement -- the only consumer of the human_verdict column.

It must never invent a verdict, must refuse to treat a stratified sample's pooled agreement as
a population estimate (that is left to the caller, and labelled), and must rejoin a blinded
annotation file to the judge's answers by idx."""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from scripts.judge_agreement import blind, render, score

FIELDS = ["idx", "dialogue_id", "direction", "reference", "candidates", "judge_verdict",
          "human_verdict", "note"]


def _write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, (j, h) in enumerate(rows):
            w.writerow({"idx": i, "dialogue_id": "9-1", "direction": "reconstruction",
                        "reference": "I am a vegetarian.", "candidates": "user: ...",
                        "judge_verdict": j, "human_verdict": h, "note": ""})
    return str(path)


def test_per_stratum_and_pooled_agreement(tmp_path):
    rows = ([("ENTAILED", "ENTAILED")] * 3 + [("ENTAILED", "PARTIAL")] * 1
            + [("ABSENT", "ABSENT")] * 4 + [("ABSENT", "ENTAILED")] * 2)
    r = score(_write(tmp_path / "m.csv", rows))
    assert r["n_filled"] == 10
    assert r["per_stratum"]["ENTAILED"] == {"n": 4, "agree": 3, "rate": pytest.approx(0.75)}
    assert r["per_stratum"]["ABSENT"] == {"n": 6, "agree": 4, "rate": pytest.approx(4 / 6)}
    assert r["sample_agreement"] == pytest.approx(0.7)
    assert r["confusion"]["ABSENT"]["ENTAILED"] == 2
    # the pooled number must be labelled as not-a-population-estimate wherever it is shown
    assert "NOT a population estimate" in render(r)


def test_unfilled_and_invalid_rows_are_reported_not_guessed(tmp_path):
    rows = [("ENTAILED", ""), ("ABSENT", "absent"), ("ABSENT", "MAYBE"), ("PARTIAL", "PARTIAL")]
    r = score(_write(tmp_path / "m.csv", rows))
    assert r["n_unfilled"] == 1
    assert r["invalid"] == [("2", "MAYBE")]
    assert r["n_filled"] == 2          # the lowercase one is accepted, MAYBE is not
    assert "still have an empty human_verdict" in render(r)


def test_credit_delta_signals_direction_of_judge_error(tmp_path):
    # judge said ABSENT (0.0) where the human said ENTAILED (1.0) -> judge under-credited
    r = score(_write(tmp_path / "m.csv", [("ABSENT", "ENTAILED")] * 2))
    assert r["mean_credit_delta"] == pytest.approx(1.0)
    r2 = score(_write(tmp_path / "m2.csv", [("ENTAILED", "ABSENT")] * 2))
    assert r2["mean_credit_delta"] == pytest.approx(-1.0)


def test_kappa_is_none_when_degenerate(tmp_path):
    r = score(_write(tmp_path / "m.csv", [("ABSENT", "ABSENT")] * 5))
    assert r["cohen_kappa"] is None      # everything in one class: kappa undefined, not 1.0
    assert r["sample_agreement"] == 1.0


def test_kappa_below_chance_is_negative(tmp_path):
    rows = [("ENTAILED", "ABSENT")] * 3 + [("ABSENT", "ENTAILED")] * 3
    assert score(_write(tmp_path / "m.csv", rows))["cohen_kappa"] < 0


def test_blind_removes_the_judge_column_and_rejoins_by_idx(tmp_path):
    src = _write(tmp_path / "m.csv", [("ENTAILED", ""), ("ABSENT", "")])
    out = blind(src)
    with open(out, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert "judge_verdict" not in rows[0], "annotator must not see the judge's answer"
    assert [r["idx"] for r in rows] == ["0", "1"]

    for r in rows:                       # annotate the blinded copy
        r["human_verdict"] = "ENTAILED"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    scored = score(out, judged_path=src)   # judge answers rejoined from the original
    assert scored["n_filled"] == 2
    assert scored["per_stratum"]["ENTAILED"]["rate"] == 1.0
    assert scored["per_stratum"]["ABSENT"]["rate"] == 0.0

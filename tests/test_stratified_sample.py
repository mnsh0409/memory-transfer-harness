"""LongMemEval's `_S` file is sorted by question_type, so `questions[:N]` is one type only.

That prefix would produce a recall number describing the easiest slice of the benchmark while
reading as though it described the benchmark. The full 500 questions carry 246,750 haystack
turns and a per-turn-ingesting system makes one model call per turn, so sampling is forced by
budget -- which is exactly the situation where a silently biased sample is most tempting.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.intrinsic_runner import stratified_question_sample


def _sorted_by_type():
    """Mirrors the real file: all of one type, then all of the next."""
    spec = [("single-session-user", 70), ("multi-session", 133),
            ("single-session-preference", 30), ("temporal-reasoning", 133),
            ("knowledge-update", 78), ("single-session-assistant", 56)]
    out = []
    for kind, n in spec:
        out += [{"question_id": f"{kind}-{i}", "question_type": kind} for i in range(n)]
    return out


def test_a_prefix_would_be_one_type_only():
    """The bug this function exists to prevent -- asserted so the motivation cannot rot."""
    qs = _sorted_by_type()
    assert len({q["question_type"] for q in qs[:30]}) == 1


def test_sample_covers_every_type():
    qs = _sorted_by_type()
    got = stratified_question_sample(qs, 30)
    assert len(got) == 30
    assert {q["question_type"] for q in got} == {q["question_type"] for q in qs}


def test_sample_is_roughly_proportional():
    qs = _sorted_by_type()
    got = stratified_question_sample(qs, 60)
    for kind in {q["question_type"] for q in qs}:
        share = sum(1 for q in qs if q["question_type"] == kind) / len(qs)
        drawn = sum(1 for q in got if q["question_type"] == kind) / len(got)
        assert abs(drawn - share) < 0.06, f"{kind}: {drawn:.3f} vs {share:.3f}"


def test_sample_is_deterministic_so_no_seed_needs_recording():
    qs = _sorted_by_type()
    assert ([q["question_id"] for q in stratified_question_sample(qs, 25)]
            == [q["question_id"] for q in stratified_question_sample(qs, 25)])


def test_sample_preserves_file_order():
    qs = _sorted_by_type()
    got = stratified_question_sample(qs, 40)
    index = {q["question_id"]: i for i, q in enumerate(qs)}
    positions = [index[q["question_id"]] for q in got]
    assert positions == sorted(positions)


def test_n_at_or_above_population_returns_everything():
    qs = _sorted_by_type()
    assert len(stratified_question_sample(qs, len(qs))) == len(qs)
    assert len(stratified_question_sample(qs, 10_000)) == len(qs)
    assert len(stratified_question_sample(qs, 0)) == len(qs)


def test_tiny_n_still_touches_every_type():
    """Under-sampling must not silently drop a stratum -- a type with zero draws would be an
    unreported gap in coverage."""
    qs = _sorted_by_type()
    got = stratified_question_sample(qs, 6)
    assert {q["question_type"] for q in got} == {q["question_type"] for q in qs}


def test_rare_stratum_is_never_over_drawn():
    qs = [{"question_id": f"a{i}", "question_type": "a"} for i in range(100)]
    qs += [{"question_id": "b0", "question_type": "b"}]
    got = stratified_question_sample(qs, 50)
    assert sum(1 for q in got if q["question_type"] == "b") == 1
    assert len(got) == 50

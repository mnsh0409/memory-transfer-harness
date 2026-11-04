"""Paper figures, scripted from collect_stats output ONLY (`make figures`): no hand-edited
charts -- every number in a figure must be reproducible by
`make figures`.

Input is runs/stats/collect_stats.json (written by scripts/collect_stats.py, which in turn
reads only what run_eval wrote). Nothing here touches run files, qrels, or the LLM cache, so
a figure can never disagree with the tables — they are the same numbers.

Three figures are computable from the extrinsic outputs alone and are built here:

  headroom   -- per collection, the frozen scope primary metric (nDCG@3) for every default-config
                run, with B0 / B1 as reference lines and each system's *delivery* annotated:
                (S - B0) / (B1 - B0), i.e. how much of the full-history headroom the memory
                system preserves. The "delivery-%" figure, restated for the
                free-data conditions (the oracle B2 is out; B1 is the ceiling).
  ablations  -- each the frozen scope ablation as a paired delta vs its un-ablated run, with the
                t-based CI from collect_stats; filled marker = Holm-significant.
  pooling    -- judged@10 beside nDCG@3 for every run (small multiples sharing the run axis,
                each with its own x-axis -- never a dual axis), the pooling-honesty panel the
                plan wants next to every nDCG column.

  cost       -- RQ1's frontier: nDCG@3 against the tokens the condition cost to produce, per
                collection, with B0/B1 marked. Needs `usage.total_tokens` on the run row
                (recorded by `harness.llm` since Aug 22); runs scored before that show as
                unknown cost and are listed rather than silently dropped.
  transfer   -- RQ3's rank-agreement scatter: each system's intrinsic score (iKAT PTKB-F1 or
                LongMemEval recall, from `runs/intrinsic_summary.json`) against its extrinsic
                nDCG@3 on CAsT. **System-level, n≈4, by design** — the extrinsic and
                intrinsic axes share no dialogue under the frozen scope (CAsT vs iKAT/LongMemEval),
                so a per-dialogue Kendall τ is not computable. The
                design already re-specified this as cross-dataset rank agreement "with
                the confound named in the same paragraph"; the caption carries that sentence.

matplotlib is an optional dependency (import-guarded like pyserini): the harness and its tests
run without it; `make_figures` raises a clear error if it is missing.
"""
from __future__ import annotations

import json
import os
import textwrap

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False

# Fixed colour per condition -- colour follows the entity, never its rank or position, so a
# figure with three systems and one with five paint Mem0 the same blue. Baselines are neutral
# inks; systems take the first five categorical slots of a CVD-validated palette.
COLORS = {
    "B0_none": "#8a8985",
    "B1_full_history": "#0b0b0b",
    "B2_gold_ptkb": "#4a3aa7",
    "S1_mem0": "#2a78d6",
    "S2_graphiti": "#eb6834",
    "S3_amem": "#1baf7a",
    "S4_letta": "#eda100",
    "S5_gam": "#e87ba4",
}
SHORT = {
    "B0_none": "B0 (none)",
    "B1_full_history": "B1 (full history)",
    "B2_gold_ptkb": "B2 (gold PTKB)",
    "S1_mem0": "Mem0",
    "S2_graphiti": "Graphiti",
    "S3_amem": "A-Mem",
    "S4_letta": "Letta",
    "S5_gam": "GAM",
}
METRIC_LABEL = {"ndcg_cut_3": "nDCG@3", "ndcg_cut_5": "nDCG@5", "recall_1000": "R@1000",
                "recip_rank": "MRR"}

COL_W = 3.35          # single-column width, inches (ACM two-column)
#: minimum B1-B0 gap for a delivery percentage to mean anything. Below this the ratio is
#: dominated by noise in the denominator; at or below zero it is not defined at all.
MIN_HEADROOM = 0.02
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _label(metric: str) -> str:
    if metric in METRIC_LABEL:
        return METRIC_LABEL[metric]
    if metric.startswith("judged_"):
        return "judged@" + metric.split("_", 1)[1]
    return metric


def _condition(label: str) -> str:
    return label.split("[", 1)[0]


def _short(label: str) -> str:
    cond, _, rest = label.partition("[")
    base = SHORT.get(cond, cond)
    return f"{base} [{rest}" if rest else base


# Candidate label placements, in the order they are tried: up-right, down-right, up-left,
# down-left. Direct labels on a scatter collide as soon as two systems land near each other,
# and "no hand-edited charts" (plan step 13) means the script has to resolve that itself.
_LABEL_OFFSETS = ((4, 3), (4, -8), (-4, 3), (-4, -8))


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _label_points(ax, items: list[tuple[float, float, str]], fontsize: float = 5.5,
                  marker_radius: float = 4.0) -> None:
    """Direct-label scatter points, giving each label the first candidate offset that doesn't
    overlap a label already placed or any marker. Deterministic: points are visited top-down
    and the offsets are tried in a fixed order, so the same data always produces the same
    figure.

    Call AFTER the axis limits are final — placement is computed in display space. Boxes are
    approximated from the character count, which is close enough to keep labels off each other
    and off the data points."""
    scale = 72.0 / ax.figure.dpi          # display px -> points, so offsets are comparable
    # Every marker is an obstacle from the outset, so a label never lands on a data point --
    # including one belonging to a different condition.
    placed: list[tuple[float, float, float, float]] = []
    for x, y, _text in items:
        px, py = (v * scale for v in ax.transData.transform((x, y)))
        placed.append((px - marker_radius, py - marker_radius,
                       px + marker_radius, py + marker_radius))
    bounds = ax.get_window_extent()
    frame = (bounds.x0 * scale, bounds.y0 * scale, bounds.x1 * scale, bounds.y1 * scale)
    for x, y, text in sorted(items, key=lambda it: (-it[1], it[0])):
        px, py = (v * scale for v in ax.transData.transform((x, y)))
        w, h = 0.55 * fontsize * len(text), 1.2 * fontsize
        candidates = []
        for dx, dy in _LABEL_OFFSETS:
            x0 = px + dx if dx > 0 else px + dx - w
            y0 = py + dy if dy > 0 else py + dy - h
            box = (x0, y0, x0 + w, y0 + h)
            # A label must stay inside the axes, or it is clipped by the figure edge or lands
            # on the tick labels. Candidates that overflow are only used as a last resort.
            inside = (box[0] >= frame[0] and box[2] <= frame[2]
                      and box[1] >= frame[1] and box[3] <= frame[3])
            candidates.append((inside, not any(_overlaps(box, o) for o in placed), dx, dy, box))
        # best available: inside and clear > inside > clear > the default up-right
        (_, _, dx, dy, box) = max(candidates, key=lambda c: (c[0] and c[1], c[0], c[1]))
        placed.append(box)
        ax.annotate(text, (x, y), fontsize=fontsize, color=INK, xytext=(dx, dy),
                    textcoords="offset points", ha="left" if dx > 0 else "right")


def _tiny(label: str) -> str:
    """Shortest unambiguous label, for figures whose panels are one column wide. `_short` keeps
    the parenthetical gloss ("B1 (full history)"), which is wider than a narrow panel and pushes
    the annotation outside the axes -- and bbox_inches="tight" then grows the whole canvas to
    contain it, shrinking every font on downscale. The gloss lives in the caption instead."""
    return _short(label).split(" (", 1)[0]


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7, length=2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def load_summary(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default_rows(summary: dict, cell: str) -> list[dict]:
    """Table rows for un-ablated default-k runs (label has no bracket), table order.

    Runs flagged `identical_to_B0` are dropped. A condition whose per-turn metrics are bitwise
    the no-memory baseline's never exercised its mechanism, and plotting it as a result is how
    a broken run ends up as the tallest bar in a headroom figure -- which is exactly what GAM's
    instrumentation failure did here before this filter existed. The same rule governs
    best-system selection in eval/stats.py; a reader is told about the run in the text, not
    shown it as a measurement."""
    rows = [r for r in summary["tables"].get(cell, []) if "[" not in r["run"]]
    return [r for r in rows if not r.get("identical_to_B0")]


def _save(fig, out_dir: str, stem: str, formats) -> list[str]:
    paths = []
    for ext in formats:
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(p)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------

def _headroom(summary: dict, cell: str, metric: str):
    """B1 - B0 on `metric` for a cell, or None when it is not positive (nothing to divide by)."""
    rows = {r["run"]: r for r in _default_rows(summary, cell)}
    b0, b1 = rows.get("B0_none"), rows.get("B1_full_history")
    if not b0 or not b1 or b0.get(metric) is None or b1.get(metric) is None:
        return None
    gap = b1[metric] - b0[metric]
    return gap if gap > 0 else None


def fig_headroom(summary: dict, out_dir: str, metric: str = "ndcg_cut_3",
                 formats=("pdf", "png")) -> list[str]:
    """Headroom bars per condition, one panel per reportable cell.

    Cells with no headroom (B1 - B0 <= 0) are omitted. That is exactly the degenerate
    history+memory design: the paper argues it cannot measure delivery, so plotting its bars
    beside the cells that can invites the reader to compare across two different experiments,
    and it spends a third of the figure on a panel whose own title says the quantity is
    undefined. The argument for the design axis lives in the text, where it can be qualified."""
    cells = sorted(c for c in summary["tables"] if _default_rows(summary, c))
    cells = [c for c in cells if _headroom(summary, c, metric) is not None] or cells
    if not cells:
        return []
    # one facet per collection, stacked vertically so the long annotations never run into a
    # neighbouring panel; x is shared so the bars are comparable across years
    n_rows = sum(len(_default_rows(summary, c)) for c in cells)
    fig, axes = plt.subplots(len(cells), 1, figsize=(COL_W, 0.28 * n_rows + 0.55 * len(cells)),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [len(_default_rows(summary, c))
                                                            for c in cells]})
    xmax = max(r[metric] for c in cells for r in _default_rows(summary, c))
    for ax, cell in zip(axes[:, 0], cells):
        rows = _default_rows(summary, cell)
        rows = list(reversed(rows))  # B0 at the bottom, systems above
        vals = [r[metric] for r in rows]
        names = [_short(r["run"]) for r in rows]
        colors = [COLORS.get(_condition(r["run"]), "#999") for r in rows]
        y = range(len(rows))
        ax.barh(y, vals, height=0.62, color=colors, linewidth=0)
        by_cond = {_condition(r["run"]): r[metric] for r in rows}
        b0, b1 = by_cond.get("B0_none"), by_cond.get("B1_full_history")
        for ref, lab in ((b0, "B0"), (b1, "B1")):
            if ref is not None:
                ax.axvline(ref, color=INK2, linewidth=0.7, linestyle=(0, (2, 2)))
        for yi, r, v in zip(y, rows, vals):
            cond = _condition(r["run"])
            txt = f"{v:.3f}"
            # The delivery fraction is (S - B0) / (B1 - B0). It is only meaningful when the
            # ceiling is actually above the floor: where B1 - B0 is ~0 or negative -- which is
            # exactly the degenerate history+memory design -- the ratio explodes and prints
            # things like "+215% of B1-B0" for a system scoring BELOW B0. Suppressed rather
            # than printed, with the facet labelled so the absence is explained.
            if cond.startswith("S") and b0 is not None and b1 is not None and (b1 - b0) > MIN_HEADROOM:
                txt += f"  ({100 * (v - b0) / (b1 - b0):+.0f}% of B1−B0)"
            marks = r.get(metric + "_marks", "")
            if marks:
                txt += " " + marks
            ax.text(v, yi, " " + txt, va="center", ha="left", fontsize=6, color=INK)
        ax.set_yticks(list(y))
        ax.set_yticklabels(names, fontsize=7, color=INK)
        ax.set_xlim(0, xmax * 1.9)   # right margin holds the annotations
        no_headroom = (b0 is not None and b1 is not None and (b1 - b0) <= MIN_HEADROOM)
        title = cell + ("   [headroom B1-B0 <= 0: delivery undefined]" if no_headroom else "")
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        _style(ax)
    axes[-1, 0].set_xlabel(_label(metric), fontsize=7)
    fig.suptitle(f"{_label(metric)}; systems annotated with share of B1−B0 headroom delivered",
                 fontsize=7, color=INK2, x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, f"fig_headroom_{metric}", formats)


def fig_ablations(summary: dict, out_dir: str, metric: str = "ndcg_cut_3",
                  formats=("pdf", "png")) -> list[str]:
    rows = [c for c in summary["comparisons"]
            if c["group"] == "ablation" and c["metric"] == metric
            and not c["collection"].startswith("pooled(") and c["mean_delta"] is not None]
    if not rows:
        return []
    rows.sort(key=lambda c: (c["treatment"], c["collection"]))
    fig, ax = plt.subplots(figsize=(COL_W, 0.42 * len(rows) + 0.9))
    for i, c in enumerate(rows):
        color = COLORS.get(_condition(c["treatment"]), "#999")
        lo, hi = c.get("ci_low"), c.get("ci_high")
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [i, i], color=color, linewidth=1.2, solid_capstyle="butt")
        filled = bool(c.get("significant"))
        ax.plot(c["mean_delta"], i, marker="o", markersize=5, color=color,
                markerfacecolor=color if filled else "white", markeredgewidth=1.2)
    ax.axvline(0, color=INK2, linewidth=0.7)
    span = max(abs(v) for c in rows for v in (c["mean_delta"], c.get("ci_low") or 0,
                                               c.get("ci_high") or 0)) or 0.01
    ax.set_xlim(-1.25 * span, 1.25 * span)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{_short(c['treatment'])}\nvs {_short(c['baseline'])} · "
                        f"{c['collection']} · n={c['n']}" for c in rows], fontsize=6, color=INK)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.set_xlabel(f"Δ {_label(metric)} (ablated − un-ablated), "
                  f"{int(round(100 * (1 - summary['alpha'])))}% CI", fontsize=7)
    ax.set_title("Ablations (filled = Holm-significant)", fontsize=8, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    return _save(fig, out_dir, f"fig_ablations_{metric}", formats)


def fig_pooling(summary: dict, out_dir: str, metric: str = "ndcg_cut_3",
                judged_metric: str | None = None, formats=("pdf", "png")) -> list[str]:
    judged_metric = judged_metric or next((m for m in summary["metrics"]
                                           if m.startswith("judged_")), None)
    if judged_metric is None:
        return []
    cells = sorted(summary["tables"])
    rows = [(cell.split("/")[0], r) for cell in cells for r in summary["tables"][cell]]
    if not rows:
        return []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_W, 0.3 * len(rows) + 0.9), sharey=True)
    names = [f"{_short(r['run'])} ({coll})" for coll, r in rows]
    y = list(range(len(rows)))[::-1]
    for ax, m in ((ax1, metric), (ax2, judged_metric)):
        for yi, (coll, r) in zip(y, rows):
            color = COLORS.get(_condition(r["run"]), "#999")
            v = r.get(m)
            if v is None:
                continue
            ax.plot([0, v], [yi, yi], color=GRID, linewidth=2, zorder=1)
            ax.plot(v, yi, marker="o", markersize=4.5, color=color, zorder=2)
        ax.set_xlabel(_label(m), fontsize=7)
        ax.set_xlim(0, 1.0 if m.startswith("judged_") else None)
        _style(ax)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=6.5, color=INK)
    fig.suptitle("Pooling honesty: judged@k next to the effectiveness metric", fontsize=7,
                 color=INK2, x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, f"fig_pooling_{metric}_{judged_metric}", formats)


def fig_cost(summary: dict, out_dir: str, metric: str = "ndcg_cut_3",
             formats=("pdf", "png")) -> list[str]:
    """RQ1: effectiveness against what the condition cost to produce.

    x is total tokens through `harness.llm` for that run — the rewrite calls plus every
    adapter call routed through the same backend.

    **Two systems make calls the harness cannot see, so their x is a LOWER BOUND:** Mem0 builds
    its own OpenAI client for extraction, and Graphiti uses its own OpenAI *embedder* for every
    ingest and search (deliberate — we evaluate systems as shipped — but uncached and
    uncounted). The caption says so rather than letting the figure imply a like-for-like
    comparison."""
    # Default configs only, via the same filter the other figures use. Plotting every ablation
    # variant put ten labels on top of each other around 1e5 tokens -- illegible once the figure
    # was drawn at a readable size. It also plotted runs flagged identical_to_B0, which the
    # headroom figure excludes: a condition whose mechanism never ran has no cost story to tell.
    # Memory-only cells when the summary has them. Panels are keyed by collection, so mixing in
    # the other context axis drew every condition twice in the same panel, one label on top of
    # the other; memory-only is also the design every extrinsic number in the paper is reported
    # under (S6). Conditional, not unconditional: a summary that never splits the context axis
    # (as the test fixtures do not) would otherwise yield no points and silently skip the figure.
    cells = sorted(summary["tables"])
    memory_only = [c for c in cells if c.endswith("/memory_only")]
    points, unknown = [], []
    for cell in (memory_only or cells):
        for r in _default_rows(summary, cell):
            tokens = (r.get("usage") or {}).get("total_tokens")
            if not tokens:
                unknown.append(r["run"])
                continue
            points.append((cell.split("/")[0], r, tokens))
    if not points:
        return []

    # This canvas is intentionally ~2.2x the width of the column it lands in, because the direct
    # point labels overflow a narrow canvas and bbox_inches="tight" then expands it anyway.
    # LaTeX scales it down by ~0.45, so every font here is specified PRE-scale: divide by CS to
    # read the printed size. At the old sizes the tick labels printed at 2.6pt.
    collections = sorted({c for c, _, _ in points})
    # Side by side, not stacked. Stacking is more legible (panels get the full column width, so
    # the canvas renders ~1:1 instead of ~0.45) but costs ~2.4in of column height against ~0.9in,
    # and the paper has no such slack -- it pushed the conclusion onto a tenth page. Revisit if
    # space frees up: the only change needed is the subplot geometry here.
    # Width is deliberately ~1.45 columns, not 2: with default configs only there are four
    # points per panel, so the panels no longer need full column width each. A narrower canvas
    # is downscaled less by LaTeX, which is what puts type back at a readable size.
    fig, axes = plt.subplots(1, len(collections), squeeze=False, sharey=True,
                             figsize=(COL_W * 1.12, 1.55))
    for ax, collection in zip(axes[0], collections):
        rows = [(r, t) for c, r, t in points if c == collection]
        for r, tokens in rows:
            cond = _condition(r["run"])
            color = COLORS.get(cond, "#999")
            ax.plot(tokens, r[metric], marker="o", markersize=5, color=color,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.set_title(collection, fontsize=8, color=INK, loc="left")
        ax.set_xscale("log")
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.margins(x=0.28, y=0.16)        # room for the direct labels, set before placing them
        _label_points(ax, [(t, r[metric], _tiny(r["run"])) for r, t in rows], fontsize=6.0)
    axes[0][0].set_ylabel(_label(metric), fontsize=7)
    for ax in axes[0]:
        ax.set_xlabel("tokens to build (log)", fontsize=7)
    caption = ("Cost is tokens through the harness backend. Lower bound for Mem0 (own client) "
               "and Graphiti (own embedder) - those calls are not counted.")
    if unknown:
        caption += f" {len(unknown)} run(s) had no usage recorded."
    # No in-figure caption. It duplicated the LaTeX caption, and as a single ~150-character line
    # it measured ~6.3in wide -- with bbox_inches="tight" the canvas grows to contain whatever is
    # drawn, so that one line, not the data, is what made this figure 7.5in wide and shrank its
    # type to 2.6pt in print. The unknown-run count it carried is stated in the LaTeX caption.
    if unknown:
        print(f"  fig_cost: {len(unknown)} run(s) without usage - state this in the caption")
    fig.tight_layout()
    return _save(fig, out_dir, f"fig_cost_{metric}", formats)


def fig_transfer(summary: dict, intrinsic: dict, out_dir: str, metric: str = "ndcg_cut_3",
                 formats=("pdf", "png")) -> list[str]:
    """RQ3: does the memory leaderboard transfer? Intrinsic score vs extrinsic nDCG@3, one
    point per SYSTEM (n is small by design — see the module docstring).

    Reports Spearman rank agreement rather than a per-dialogue Kendall τ, because the two axes
    share no dialogue. The confound sentence the plan requires is in the caption."""
    extrinsic = {}
    for cell in summary["tables"]:
        if not cell.startswith("cast19"):
            continue
        for r in summary["tables"][cell]:
            if "[" not in r["run"] and r.get(metric) is not None:
                extrinsic[_condition(r["run"])] = r[metric]

    axes_spec = [("ptkb_reconstruction", "mean_f1", "iKAT PTKB F1"),
                 ("longmemeval_recall", "mean_evidence_recall", "LongMemEval recall")]
    panels, n_unvalidated = [], 0
    for axis_name, key, label in axes_spec:
        pts = []
        for row in intrinsic.values():
            if row.get("axis") != axis_name or row.get("ablation"):
                continue
            # An UNVALIDATED row comes from --lexical-judge, a plumbing stand-in for the frozen
            # LLM judge. Its numbers are meaningless and run_intrinsic labels them as such; a
            # figure that plotted them anyway would launder a smoke run into a result. Same
            # rule as run_eval refusing native metrics for reported numbers.
            if "UNVALIDATED" in str(row.get("judge_source", "")):
                n_unvalidated += 1
                continue
            cond = row.get("condition")
            if cond in extrinsic and row.get(key) is not None:
                pts.append((cond, row[key], extrinsic[cond]))
        if pts:
            panels.append((label, sorted(pts)))
    if not panels:
        return []

    # a two-panel figure spans both columns in the ACM template (figure*), so it gets the
    # full text width rather than being squeezed into one column
    fig, axes = plt.subplots(1, len(panels), squeeze=False, sharey=True,
                             figsize=(COL_W * len(panels), 2.3))
    for i, (ax, (label, pts)) in enumerate(zip(axes[0], panels)):
        for cond, x, y in pts:
            ax.plot(x, y, marker="o", markersize=6, color=COLORS.get(cond, "#999"),
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        rho = _spearman([x for _, x, _ in pts], [y for _, _, y in pts])
        rho_text = "n/a" if rho is None else f"{rho:+.2f}"
        # n and rho ride on the x-axis label: as a title they are long enough that two panels
        # at single-column width overlap, and inside the panel they collide with the points.
        # The axis label already names the measure, so there is no separate title to repeat it.
        ax.set_xlabel(f"{label}\n(n={len(pts)}, Spearman ρ={rho_text})", fontsize=7)
        if i == 0:
            ax.set_ylabel(f"CAsT 2019 {_label(metric)}", fontsize=7)
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.margins(x=0.26, y=0.20)
        _label_points(ax, [(x, y, SHORT.get(cond, cond)) for cond, x, y in pts], fontsize=6)
    caption = ("Rank agreement across datasets. Disagreement may reflect dataset difference, "
               "not benchmark invalidity.")
    if n_unvalidated:
        caption += f" ({n_unvalidated} unvalidated-judge row(s) excluded.)"
    fig.suptitle(caption, fontsize=6, color=INK2, x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, f"fig_transfer_{metric}", formats)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman ρ (Pearson on ranks, average ranks for ties). None when undefined."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(vs):
        order = sorted(range(n), key=lambda i: vs[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


# ---------------------------------------------------------------------------

def make_figures(stats_dir: str = os.path.join("runs", "stats"),
                 out_dir: str = os.path.join("runs", "figures"),
                 metric: str = "ndcg_cut_3", formats=("pdf", "png"),
                 intrinsic_path: str | None = None) -> dict:
    """Build every figure from {stats_dir}/collect_stats.json into out_dir. Regenerated in
    place (pure function of collect_stats output). Returns {"figures": {...}, "manifest": path}."""
    if not HAVE_MPL:
        raise RuntimeError("matplotlib is not installed -- `pip install matplotlib` "
                           "(figures are an optional dependency; nothing else needs it)")
    summary_path = os.path.join(stats_dir, "collect_stats.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"{summary_path} not found -- run scripts/collect_stats first")
    summary = load_summary(summary_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    intrinsic_path = intrinsic_path or os.path.join(
        os.path.dirname(os.path.dirname(summary_path)), "intrinsic_summary.json")
    intrinsic = {}
    if os.path.exists(intrinsic_path):
        with open(intrinsic_path, encoding="utf-8") as f:
            intrinsic = json.load(f)

    figures = {
        "headroom": fig_headroom(summary, out_dir, metric, formats),
        "ablations": fig_ablations(summary, out_dir, metric, formats),
        "pooling": fig_pooling(summary, out_dir, metric, formats=formats),
        "cost": fig_cost(summary, out_dir, metric, formats),
        "transfer": fig_transfer(summary, intrinsic, out_dir, metric, formats),
    }
    manifest = {
        "source": summary_path,
        "source_stats_summary": summary.get("source"),
        "source_intrinsic_summary": intrinsic_path if intrinsic else None,
        "metric": metric,
        "n_runs": summary.get("n_runs"),
        "figures": figures,
        "skipped": [k for k, v in figures.items() if not v],
        "skip_reasons": {
            "cost": "no run row carries usage.total_tokens (scored before token accounting, "
                    "or an echo-backend run)",
            "transfer": "needs runs/intrinsic_summary.json rows whose condition also has a "
                        "cast19 extrinsic row",
        },
    }
    manifest_path = os.path.join(out_dir, "figures_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return {"figures": figures, "manifest": manifest_path, "skipped": manifest["skipped"]}

"""
Quality + Growth combination test: which pairings of ranked metrics (".R")
best line up with the ranked 1-year forward return (".R.Y") in PanelPrep's
panel, at most two metrics per category.

Each combination's score is the average of its Quality metrics' buckets and
its Growth metrics' buckets, each category weighted equally (1 = best). Its
IC is the Spearman correlation between that score and the forward-return
bucket each quarter, as in factor_ic.py.

Combinations are chosen on a training window and checked on a later test
window. 1-year returns overlap, so the test window starts GAP_QUARTERS after
the training window ends: no test-quarter return window overlaps a training
one. Growth metrics configured as "Lower" (investment, capex, asset growth)
are also tried flipped ("~" prefix), as factor_ic showed they predicted in
the other direction in Technology.

Run from the project root:
    python reports/combo_ic.py
Writes CSVs to reports/<sector>_combo_ic_<end period>/ (CSV files are
git-ignored; rerun to regenerate them).
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from factor_ic import nw_t  # noqa: E402
from metrics_cfg import load_metrics_cfg  # noqa: E402
from panel_prep import PanelPrep  # noqa: E402

SECTORS = ["Technology"]
END_PERIOD = "2025Q2"
TRAIN_END = "2020Q4"
GAP_QUARTERS = 3
MAX_PER_CATEGORY = 2
TOP_N = 25


def spearman_rows(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of `scores` (combos x companies) with y."""
    rs = rankdata(scores, axis=1)
    ry = rankdata(y)
    rs -= rs.mean(axis=1, keepdims=True)
    ry -= ry.mean()
    return (rs @ ry) / np.sqrt((rs ** 2).sum(axis=1) * (ry @ ry))


def main() -> None:
    prep = PanelPrep(sectors=SECTORS, end_period=END_PERIOD)
    d = prep.prepare()
    y_col = f"{prep.target.removesuffix('.Y')}.R.Y"
    cfg = load_metrics_cfg()["Metrics"]
    directions = prep.direction_map()

    buckets = {m: d[f"{m}.R"] for m in cfg["Quality"]}
    growth = list(cfg["Growth"])
    for m in cfg["Growth"]:
        buckets[m] = d[f"{m}.R"]
        if directions[m] == "Lower":
            buckets[f"~{m}"] = prep.n_buckets + 1 - d[f"{m}.R"]
            growth.append(f"~{m}")

    def subsets(names):
        return [c for k in range(1, MAX_PER_CATEGORY + 1) for c in combinations(names, k)
                if not (len(c) == 2 and c[0].lstrip("~") == c[1].lstrip("~"))]
    q_sets, g_sets = subsets(list(cfg["Quality"])), subsets(growth)
    combos = [(q, g) for q in q_sets for g in g_sets]

    # Category scores per subset, then each combo = (quality score + growth score) / 2.
    b = pd.DataFrame(buckets).to_numpy(float)
    col = {m: i for i, m in enumerate(buckets)}
    q_score = np.stack([b[:, [col[m] for m in s]].mean(axis=1) for s in q_sets])
    g_score = np.stack([b[:, [col[m] for m in s]].mean(axis=1) for s in g_sets])
    qi = np.repeat(np.arange(len(q_sets)), len(g_sets))
    gi = np.tile(np.arange(len(g_sets)), len(q_sets))

    period = d.index.get_level_values("Period")
    periods = sorted(period.unique())
    y = d[y_col].to_numpy(float)
    ic = np.empty((len(combos), len(periods)))
    for j, p in enumerate(periods):
        rows = np.flatnonzero(period == p)
        # Higher IC = better: bucket 1 is best on both sides, so rank both ascending.
        ic[:, j] = spearman_rows(q_score[:, rows][qi] + g_score[:, rows][gi], y[rows])

    train = np.array([p <= pd.Period(TRAIN_END, "Q") for p in periods])
    test = np.array([p > pd.Period(TRAIN_END, "Q") + GAP_QUARTERS for p in periods])
    ic_df = pd.DataFrame(ic, columns=[str(p) for p in periods])
    out = pd.DataFrame({
        "quality": [" + ".join(q) for q, _ in combos],
        "growth": [" + ".join(g) for _, g in combos],
        "n_metrics": [len(q) + len(g) for q, g in combos],
        "train_ic": ic[:, train].mean(axis=1),
        "train_t": ic_df.loc[:, train].apply(nw_t, axis=1),
        "test_ic": ic[:, test].mean(axis=1),
        "test_t": ic_df.loc[:, test].apply(nw_t, axis=1),
        "test_pct_pos": (ic[:, test] > 0).mean(axis=1),
        "full_ic": ic.mean(axis=1),
    }).sort_values("train_t", ascending=False)

    # Single-metric baselines on the same windows (a combination should beat these).
    singles = []
    for m in list(cfg["Quality"]) + growth:
        s = pd.Series([pd.Series(b[period == p, col[m]]).corr(pd.Series(y[period == p]), method="spearman")
                       for p in periods])
        singles.append({"metric": m, "train_ic": s[train].mean(), "train_t": nw_t(s[train]),
                        "test_ic": s[test].mean(), "test_t": nw_t(s[test])})
    singles = pd.DataFrame(singles).sort_values("train_t", ascending=False)

    # How often each metric appears in the top 1% of combinations by training t-stat.
    top = out.head(max(1, len(out) // 100))
    freq = (pd.concat([top["quality"], top["growth"]]).str.split(" \\+ ").explode()
            .value_counts().rename("top1pct_count").rename_axis("metric").reset_index())

    test_periods = [str(p) for p, t in zip(periods, test) if t]
    window = pd.DataFrame([{
        "train": f"{periods[0]}-{TRAIN_END}", "test": f"{test_periods[0]}-{test_periods[-1]}",
        "train_quarters": int(train.sum()), "test_quarters": int(test.sum()),
        "combinations": len(combos), "quality_candidates": len(cfg["Quality"]),
        "growth_candidates": len(growth),
    }])

    dest = Path(__file__).resolve().parent / f"{'_'.join(SECTORS).lower()}_combo_ic_{END_PERIOD}"
    dest.mkdir(exist_ok=True)
    out.round(4).to_csv(dest / "combos.csv", index=False)
    singles.round(4).to_csv(dest / "singles.csv", index=False)
    freq.to_csv(dest / "top1pct_metrics.csv", index=False)
    window.to_csv(dest / "window.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)
    print(window.to_string(index=False))
    print(f"\nTop {TOP_N} combinations by training t-stat:")
    print(out.head(TOP_N).round(3).to_string(index=False))
    print("\nTest IC across all combinations: "
          f"median {np.median(out['test_ic']):.3f}, top-{TOP_N} median {out.head(TOP_N)['test_ic'].median():.3f}")
    print("\nSingle metrics (top 10 by training t):")
    print(singles.head(10).round(3).to_string(index=False))
    print("\nMetrics most often in the top 1% of combinations:")
    print(freq.head(15).to_string(index=False))
    print(f"\nWrote {dest.relative_to(ROOT)}/ (combos, singles, top1pct_metrics, window)")


if __name__ == "__main__":
    main()

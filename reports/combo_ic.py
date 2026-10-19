"""
Quality + Growth combination test: which pairings of ranked metrics (".R")
best line up with the ranked 1-year forward return (".R.Y") in PanelPrep's
panel, at most two metrics per category.

Each combination's score is the average of its Quality metrics' buckets and
its Growth metrics' buckets, each category weighted equally (1 = best). Its
IC is the Spearman correlation between that score and the forward-return
bucket each quarter, as in factor_ic.py.

1-year returns overlap, so every train/test split leaves GAP_QUARTERS between
the last training quarter and the first test quarter: no test return window
overlaps a training one. Growth metrics configured as "Lower" (investment,
capex, asset growth) are also tried flipped ("~" prefix), as factor_ic showed
they predicted in the other direction in Technology.

Selection, on any window: rank by Newey-West t-stat, skipping combinations
with two metrics of the same family in one category (e.g. ROE + ROE_3Yr,
Payout_Ratio + FCF_Payout_Ratio, and any two investment metrics such as
Inv_Rate_3Yr + ~Asset_G_Rate), and combinations
that don't beat every smaller model they contain (nested rule: paired
Newey-West t of the quarterly IC difference above NESTED_T in that window).
Raw t-stats are also shown Bonferroni-adjusted for the number of models of
that size searched.

Three uses of that selection:
  - Holdout: pick on TRAIN_END and earlier, report the later test window.
  - Walk-forward: train on everything up to each year-end from WF_FIRST_TRAIN_END,
    test the next WF_STEP quarters after the gap, roll forward a year. The
    out-of-sample IC of each size's pick, strung together, chooses model size.
  - Final: pick on the full sample (no holdout left; walk-forward results
    stand in for their expected performance).

Run from the project root:
    python reports/combo_ic.py
Writes CSVs to reports/<sector>_combo_ic_<end period>/ (CSV files are
git-ignored; rerun to regenerate them).
"""
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from factor_ic import NW_LAGS, nw_t  # noqa: E402
from metrics_cfg import load_metrics_cfg  # noqa: E402
from panel_prep import PanelPrep  # noqa: E402

SECTORS = ["Technology"]
END_PERIOD = "2025Q2"
TRAIN_END = "2020Q4"
GAP_QUARTERS = 3
MAX_PER_CATEGORY = 2
TOP_N = 25
TOP_TABLE = 200
CANDIDATES = 3            # per model size
NESTED_T = 0.0            # a model must beat each smaller model it contains by this paired t
NESTED_T_ALTERNATIVES = (2.0, None)  # walk-forward also run with these (None = no nested rule)
WF_FIRST_TRAIN_END = "2015Q4"
WF_STEP = 4               # walk-forward test block and roll, in quarters
# Models chosen for use after reviewing the results (2026-10-18); reported in chosen.csv.
CHOSEN = {
    "Top choice": "FCF_Payout_Ratio | CFOPS_CAGR_3Yr + ~CapX_Intensity_3Yr",
    "Second choice": "FCF_Payout_Ratio | FCFPS_CAGR_3Yr + ~CapX_Intensity_3Yr",
}
ERAS = [("2011Q4-2015Q4", "2015Q4"), ("2016Q1-2020Q4", "2020Q4"), ("2021Q1-2025Q2", END_PERIOD)]


def spearman_rows(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of `scores` (combos x companies) with y."""
    rs = rankdata(scores, axis=1)
    ry = rankdata(y)
    rs -= rs.mean(axis=1, keepdims=True)
    ry -= ry.mean()
    return (rs @ ry) / np.sqrt((rs ** 2).sum(axis=1) * (ry @ ry))


def nw_t_rows(x: np.ndarray, lags: int = NW_LAGS) -> np.ndarray:
    """factor_ic.nw_t of each row of x (rows x quarters, no missing values)."""
    n = x.shape[1]
    e = x - x.mean(axis=1, keepdims=True)
    s = (e * e).sum(axis=1) / n
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1)) * (e[:, lag:] * e[:, :-lag]).sum(axis=1) / n
    return x.mean(axis=1) / np.sqrt(s / n)


def bonferroni_t(t: np.ndarray, m: np.ndarray) -> np.ndarray:
    """One-sided t-stat after multiplying its p-value by the m models searched (floored at 0)."""
    return np.maximum(norm.isf(np.minimum(1.0, norm.sf(t) * m)), 0.0)


# Metrics that all measure how fast a company invests; a model may hold only one of them.
INVESTMENT_FAMILY = ("Inv_Rate", "CapX_Intensity", "Asset_G_Rate")


def family(m: str) -> str:
    """What a metric measures: drops the "~" flip, the horizon and an FCF/CFO prefix;
    investment metrics share one family."""
    base = re.sub(r"^(FCF|CFO)(PS)?_|_(1|3|5)Yr$", "", m.lstrip("~"))
    return "Investment" if base in INVESTMENT_FAMILY else base


def overlaps(s: tuple) -> bool:
    return len(s) == 2 and family(s[0]) == family(s[1])


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
            # Flip actual ranks only: a filled rank (middle, or bottom under Bottom_If) stays as filled.
            buckets[f"~{m}"] = d[f"{m}.R"].where(d[f"{m}.F"].eq(1), prep.n_buckets + 1 - d[f"{m}.R"])
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
    assert not np.isnan(ic).any(), "a quarter's IC is undefined"

    n_metrics = np.array([len(q) + len(g) for q, g in combos])
    overlap = np.array([overlaps(q) or overlaps(g) for q, g in combos])
    # Models searched per size (same-family pairs are ruled out before looking at results).
    pool = {k: int(((n_metrics == k) & ~overlap).sum()) for k in (2, 3, 4)}
    pool_of = np.array([pool[k] for k in n_metrics])

    # Nested rule: each (model, smaller model it contains) pair, dropping one metric
    # from a category that has two.
    index = {c: i for i, c in enumerate(combos)}
    parent, child = [], []
    for i, (q, g) in enumerate(combos):
        subs = ([((m,), g) for m in q] if len(q) == 2 else []) + ([(q, (m,)) for m in g] if len(g) == 2 else [])
        for s in subs:
            parent.append(i)
            child.append(index[s])
    parent, child = np.array(parent), np.array(child)

    def select(window: np.ndarray):
        """Training t, worst paired t against a contained model, and eligibility on `window`."""
        t = nw_t_rows(ic[:, window])
        sub_t = np.full(len(combos), np.inf)
        for k in range(0, len(parent), 50_000):
            p, c = parent[k:k + 50_000], child[k:k + 50_000]
            np.minimum.at(sub_t, p, nw_t_rows(ic[p][:, window] - ic[c][:, window]))
        return t, sub_t, ~overlap & (sub_t > NESTED_T)

    def top(t, eligible, k, n):
        idx = np.flatnonzero(eligible & (n_metrics == k))
        return idx[np.argsort(-t[idx])[:n]]

    pidx = pd.PeriodIndex(periods)
    train_end = pd.Period(TRAIN_END, "Q")
    train = np.asarray(pidx <= train_end)
    test = np.asarray(pidx > train_end + GAP_QUARTERS)
    full = np.ones(len(periods), bool)
    train_t, train_sub_t, train_ok = select(train)
    full_t, full_sub_t, full_ok = select(full)

    out = pd.DataFrame({
        "quality": [" + ".join(q) for q, _ in combos],
        "growth": [" + ".join(g) for _, g in combos],
        "n_metrics": n_metrics,
        "overlap": overlap,
        # Contrast: an investment metric in both directions (measures whether investment is speeding up).
        "contrast": [overlaps(g) and g[0].startswith("~") != g[1].startswith("~") for _, g in combos],
        "nested_ok": train_ok,
        "min_sub_t": np.where(np.isinf(train_sub_t), np.nan, train_sub_t),
        "train_ic": ic[:, train].mean(axis=1),
        "train_t": train_t,
        "train_t_adj": bonferroni_t(train_t, pool_of),
        "test_ic": ic[:, test].mean(axis=1),
        "test_t": nw_t_rows(ic[:, test]),
        "test_pct_pos": (ic[:, test] > 0).mean(axis=1),
        "full_ic": ic.mean(axis=1),
        "full_t": full_t,
        "full_t_adj": bonferroni_t(full_t, pool_of),
        "full_nested_ok": full_ok,
    }).sort_values("train_t", ascending=False)
    out.insert(0, "train_rank", np.arange(1, len(out) + 1))
    out.insert(1, "combo", out["quality"] + " | " + out["growth"])
    label = np.where(train, "Train", np.where(test, "Test", "Gap"))

    def picks(t, eligible, cols):
        idx = [i for k in (2, 3, 4) for i in top(t, eligible, k, CANDIDATES)]
        p = out.loc[idx, cols].copy()
        p.insert(0, "candidate", [f"{k}{chr(ord('A') + j)}" for k in (2, 3, 4)
                                  for j in range(len(top(t, eligible, k, CANDIDATES)))])
        return p, idx

    # Holdout candidates: picked on training, judged on test.
    cand, cand_idx = picks(train_t, train_ok, [
        "train_rank", "combo", "quality", "growth", "n_metrics", "min_sub_t", "train_ic", "train_t", "train_t_adj",
        "test_ic", "test_t", "test_pct_pos"])
    cand_q = pd.DataFrame([{"period": str(p), "window": label[j], "candidate": c, "ic": ic[i, j]}
                           for c, i in zip(cand["candidate"], cand_idx) for j, p in enumerate(periods)])

    # Final models: picked on the full sample.
    final, final_idx = picks(full_t, full_ok, ["combo", "quality", "growth", "n_metrics", "full_ic", "full_t",
                                               "full_t_adj"])
    final["full_sub_t"] = np.where(np.isinf(full_sub_t[final_idx]), np.nan, full_sub_t[final_idx])
    final["full_pct_pos"] = (ic[final_idx] > 0).mean(axis=1)
    final_q = pd.DataFrame([{"period": str(p), "candidate": c, "ic": ic[i, j]}
                            for c, i in zip(final["candidate"], final_idx) for j, p in enumerate(periods)])

    # Chosen models: full-sample, holdout and per-era results.
    chosen_idx = [out.index[out["combo"] == c][0] for c in CHOSEN.values()]
    rank3 = out[(out["n_metrics"] == 3) & out["full_nested_ok"]].sort_values("full_t", ascending=False)["combo"].tolist()
    chosen = out.loc[chosen_idx, ["combo", "quality", "growth", "n_metrics", "full_ic", "full_t", "full_t_adj",
                                  "train_rank", "train_ic", "train_t", "test_ic", "test_t", "test_pct_pos"]].copy()
    chosen.insert(0, "role", list(CHOSEN))
    chosen["full_rank_size"] = [rank3.index(c) + 1 if c in rank3 else np.nan for c in chosen["combo"]]
    chosen["full_sub_t"] = full_sub_t[chosen_idx]
    chosen["full_pct_pos"] = (ic[chosen_idx] > 0).mean(axis=1)
    era_start = pidx[0]
    for name, end in ERAS:
        era = np.asarray((pidx >= era_start) & (pidx <= pd.Period(end, "Q")))
        chosen[f"ic_{name}"] = ic[chosen_idx][:, era].mean(axis=1)
        era_start = pd.Period(end, "Q") + 1
    chosen["worst_4q_ic"] = [pd.Series(ic[i]).rolling(4).mean().min() for i in chosen_idx]
    chosen_q = pd.DataFrame([{"period": str(p), "role": r, "ic": ic[i, j]}
                             for r, i in zip(CHOSEN, chosen_idx) for j, p in enumerate(periods)])

    # Walk-forward: expanding training window, one-year test blocks after the gap. Run
    # for NESTED_T and each alternative threshold (None = no nested rule) so the
    # rule's effect on out-of-sample results can be compared.
    rules = [(f"t > {NESTED_T:g}", NESTED_T)] + [
        ("None" if r is None else f"t > {r:g}", -np.inf if r is None else r) for r in NESTED_T_ALTERNATIVES]
    folds, wf_rows, wf_picks = [], [], []
    t_end = pd.Period(WF_FIRST_TRAIN_END, "Q")
    while t_end + GAP_QUARTERS + 1 <= pidx[-1]:
        tr = np.asarray(pidx <= t_end)
        te = np.asarray((pidx > t_end + GAP_QUARTERS) & (pidx <= t_end + GAP_QUARTERS + WF_STEP))
        t, sub_t, _ = select(tr)
        fold = len(folds) + 1
        folds.append({"fold": fold, "train": f"{pidx[0]}-{t_end}",
                      "test": f"{pidx[te][0]}-{pidx[te][-1]}", "test_quarters": int(te.sum())})
        for rule, thr in rules:
            ok = ~overlap & (sub_t > thr)
            for k in (2, 3, 4):
                top_n = top(t, ok, k, TOP_N)
                if not len(top_n):
                    continue
                best = top_n[0]
                wf_picks.append({"rule": rule, "fold": fold, "train": folds[-1]["train"], "test": folds[-1]["test"],
                                 "n_metrics": k, "combo": out.at[best, "combo"],
                                 "train_t": t[best], "test_ic": ic[best, te].mean()})
                for j in np.flatnonzero(te):
                    wf_rows.append({"rule": rule, "period": str(pidx[j]), "fold": fold, "n_metrics": k,
                                    "ic_pick": ic[best, j], "ic_top": ic[top_n, j].mean()})
        t_end += WF_STEP
    folds, wf_picks, wf_q = pd.DataFrame(folds), pd.DataFrame(wf_picks), pd.DataFrame(wf_rows)
    sizes = []
    for (rule, thr), r_picks in zip(rules, (wf_picks[wf_picks["rule"] == r] for r, _ in rules)):
        fold_wins = r_picks.loc[r_picks.groupby("fold")["test_ic"].idxmax(), "n_metrics"].value_counts()
        tr_ok, fu_ok = ~overlap & (train_sub_t > thr), ~overlap & (full_sub_t > thr)
        for k in (2, 3, 4):
            g = wf_q[(wf_q["rule"] == rule) & (wf_q["n_metrics"] == k)]
            tr_top = top(train_t, tr_ok, k, TOP_N)
            sizes.append({
                "rule": rule, "n_metrics": k, "models": pool[k],
                "train_eligible": int((tr_ok & (n_metrics == k)).sum()),
                "full_eligible": int((fu_ok & (n_metrics == k)).sum()),
                "wf_folds_picked": int(g["fold"].nunique()),
                "wf_pick_ic": g["ic_pick"].mean() if len(g) else np.nan,
                "wf_pick_t": nw_t(g["ic_pick"]) if len(g) > 4 else np.nan,
                "wf_pick_pct_pos": (g["ic_pick"] > 0).mean() if len(g) else np.nan,
                "wf_top_ic": g["ic_top"].mean() if len(g) else np.nan,
                "wf_top_t": nw_t(g["ic_top"]) if len(g) > 4 else np.nan,
                "wf_folds_won": int(fold_wins.get(k, 0)),
                "holdout_top_test_ic": ic[tr_top][:, test].mean() if len(tr_top) else np.nan,
            })
    sizes = pd.DataFrame(sizes)
    # Recommended size per rule: highest out-of-sample IC among sizes picked in every fold.
    full_cover = sizes["wf_folds_picked"] == len(folds)
    best_ic = sizes[full_cover].groupby("rule")["wf_pick_ic"].transform("max")
    sizes["recommended"] = np.where(full_cover & (sizes["wf_pick_ic"] == best_ic.reindex(sizes.index)), "Yes", "No")

    # Single-metric baselines on the same windows (a combination should beat these).
    singles = []
    for m in list(cfg["Quality"]) + growth:
        s = pd.Series([pd.Series(b[period == p, col[m]]).corr(pd.Series(y[period == p]), method="spearman")
                       for p in periods])
        singles.append({"metric": m, "train_ic": s[train].mean(), "train_t": nw_t(s[train]),
                        "test_ic": s[test].mean(), "test_t": nw_t(s[test])})
    singles = pd.DataFrame(singles).sort_values("train_t", ascending=False)

    # How often each metric appears in the top 1% of combinations by training t-stat.
    top1 = out.head(max(1, len(out) // 100))
    freq = (pd.concat([top1["quality"], top1["growth"]]).str.split(" \\+ ").explode()
            .value_counts().rename("top1pct_count").rename_axis("metric").reset_index())

    deciles = (out.assign(train_decile=pd.qcut(out["train_ic"], 10, labels=range(1, 11)))
               .groupby("train_decile", observed=True)
               .agg(train_ic_median=("train_ic", "median"), test_ic_median=("test_ic", "median")).reset_index())
    best_single = singles.iloc[0]
    eligible = out[out["nested_ok"]]
    selection = pd.DataFrame([
        {"group": "All combinations", "test_ic_median": out["test_ic"].median()},
        {"group": f"Top {TOP_N} by training t", "test_ic_median": out.head(TOP_N)["test_ic"].median()},
        {"group": f"Top {TOP_N}, selection rules applied", "test_ic_median": eligible.head(TOP_N)["test_ic"].median()},
        {"group": f"Best single metric ({best_single['metric']})", "test_ic_median": best_single["test_ic"]},
    ])

    test_periods = [str(p) for p, t in zip(periods, test) if t]
    window = pd.DataFrame([{
        "sample": f"{periods[0]}-{periods[-1]}", "quarters": len(periods), "rows": len(d),
        "companies": d.index.get_level_values("Ticker").nunique(),
        "train": f"{periods[0]}-{TRAIN_END}", "test": f"{test_periods[0]}-{test_periods[-1]}",
        "train_quarters": int(train.sum()), "test_quarters": int(test.sum()),
        "combinations": len(combos), "quality_candidates": len(cfg["Quality"]),
        "growth_candidates": len(growth), "nested_t": NESTED_T,
        "rule": rules[0][0], "wf_folds": len(folds), "wf_test": f"{folds['test'].iloc[0][:6]}-{folds['test'].iloc[-1][-6:]}",
        "wf_test_quarters": int(folds["test_quarters"].sum()),
    }])

    yes_no = {True: "Yes", False: "No"}
    top_table = (out.head(TOP_TABLE).drop(columns=["full_ic", "full_t", "full_t_adj", "full_nested_ok"])
                 .replace({"overlap": yes_no, "contrast": yes_no, "nested_ok": yes_no}))

    dest = Path(__file__).resolve().parent / f"{'_'.join(SECTORS).lower()}_combo_ic_{END_PERIOD}"
    dest.mkdir(exist_ok=True)
    for f in dest.glob("*.csv"):
        f.unlink()  # outputs from older runs shouldn't linger
    out.round(4).to_csv(dest / "combos.csv", index=False)
    top_table.round(4).to_csv(dest / "top_combos.csv", index=False)
    cand.round(4).to_csv(dest / "candidates.csv", index=False)
    cand_q.round(4).to_csv(dest / "candidate_quarterly.csv", index=False)
    final.round(4).to_csv(dest / "final_models.csv", index=False)
    final_q.round(4).to_csv(dest / "final_quarterly.csv", index=False)
    chosen.round(4).to_csv(dest / "chosen.csv", index=False)
    chosen_q.round(4).to_csv(dest / "chosen_quarterly.csv", index=False)
    sizes.round(4).to_csv(dest / "size_summary.csv", index=False)
    folds.to_csv(dest / "wf_folds.csv", index=False)
    wf_picks.round(4).to_csv(dest / "wf_picks.csv", index=False)
    wf_q.round(4).to_csv(dest / "wf_quarterly.csv", index=False)
    singles.round(4).to_csv(dest / "singles.csv", index=False)
    freq.to_csv(dest / "top1pct_metrics.csv", index=False)
    deciles.round(4).to_csv(dest / "deciles.csv", index=False)
    selection.round(4).to_csv(dest / "selection_summary.csv", index=False)
    window.to_csv(dest / "window.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 60)
    print(window.T.to_string(header=False))
    print(f"\nTop {TOP_N} combinations by training t-stat:")
    print(out.head(TOP_N)[["train_rank", "combo", "n_metrics", "overlap", "nested_ok", "train_ic", "train_t",
                           "train_t_adj", "test_ic", "test_t"]].round(3).to_string(index=False))
    print("\nHoldout candidates (picked on training):")
    print(cand.drop(columns=["quality", "growth"]).round(3).to_string(index=False))
    print("\nFinal models (picked on the full sample):")
    print(final.drop(columns=["quality", "growth"]).round(3).to_string(index=False))
    print("\nChosen models:")
    print(chosen.drop(columns=["quality", "growth"]).round(3).T.to_string(header=False))
    print("\nWalk-forward picks:")
    print(wf_picks.round(3).to_string(index=False))
    print("\nModel size:")
    print(sizes.round(3).to_string(index=False))
    print("\nSelection groups:")
    print(selection.round(3).to_string(index=False))
    print("\nSingle metrics (top 10 by training t):")
    print(singles.head(10).round(3).to_string(index=False))
    print(f"\nWrote {dest.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

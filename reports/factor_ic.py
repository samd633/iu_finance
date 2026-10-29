"""
Information-coefficient (IC) report: how each ranked metric (".R") lines up
with the ranked 1-year forward return (".R.Y") in PanelPrep's panel.

For every metric, each quarter: Spearman correlation between the metric's
bucket and the forward-return bucket (both 1 = best), averaged across
quarters. 1-year returns sampled quarterly overlap, so t-stats use Newey-West
standard errors with 3 lags. Also reports the share of positive quarters, the
IC on actual (unfilled) ranks only, and the median forward return by bucket.

Run from the project root:
    python reports/factor_ic.py
for Technology; set REPORT_SECTOR for another sector (PowerShell:
$env:REPORT_SECTOR = "Healthcare"; bash: REPORT_SECTOR=Healthcare python ...).
The same variable selects the sector in combo_ic.py and addon_ic.py.
Writes CSVs to reports/<sector>_factor_ic_<end period>/ (CSV files are
git-ignored; rerun to regenerate them).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics_cfg import load_metrics_cfg  # noqa: E402
from panel_prep import PanelPrep  # noqa: E402

SECTORS = [os.environ.get("REPORT_SECTOR", "Technology")]
END_PERIOD = "2025Q2"
NW_LAGS = 3
SIG_T = 2.0


def nw_t(x: pd.Series, lags: int = NW_LAGS) -> float:
    """Newey-West t-stat of the mean of x."""
    x = x.dropna().to_numpy(float)
    n, e = len(x), x - x.mean()
    s = e @ e / n
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return x.mean() / np.sqrt(s / n)


def main() -> None:
    prep = PanelPrep(sectors=SECTORS, end_period=END_PERIOD)
    d = prep.prepare()
    y = f"{prep.target.removesuffix('.Y')}.R.Y"
    cfg = load_metrics_cfg()["Metrics"]
    group_of = {n: g for g, ms in cfg.items() for n in ms}
    directions, rules = prep.direction_map(), prep.bottom_if_map()
    metrics = [c[:-2] for c in d if c.endswith(".R")]
    period = d.index.get_level_values("Period")
    per = d.groupby(level="Period")

    rows, quarterly = [], []
    for m in metrics:
        r, f = f"{m}.R", f"{m}.F"
        ic = per.apply(lambda g: g[r].corr(g[y], method="spearman"))
        ic_actual = per.apply(lambda g: g.loc[g[f] == 0, r].corr(g.loc[g[f] == 0, y], method="spearman"))
        # Median (not mean) return per bucket: a few +500% winners would dominate means.
        by_bucket = d.groupby([period, d[r].round()])[prep.target].median().unstack()
        q = by_bucket.mean().reindex([1.0, 2.0, 3.0, 4.0, 5.0])
        rows.append({
            "metric": m, "group": group_of[m], "direction": directions[m], "bottom_if": rules.get(m, ""),
            "mean_ic": ic.mean(), "nw_t": nw_t(ic), "pct_pos": (ic > 0).mean(),
            "mean_ic_actual": ic_actual.mean(), "pooled": d[r].corr(d[y], method="spearman"),
            "spread_q1_q5": (by_bucket[1.0] - by_bucket[5.0]).mean(), "filled_pct": d[f].mean(),
            **{f"q{i}_median_ret": q[float(i)] for i in range(1, 6)},
        })
        quarterly += [{"period": str(p), "metric": m, "ic": v} for p, v in ic.items()]
    summary = pd.DataFrame(rows).sort_values("mean_ic", ascending=False)

    sig = summary.loc[summary["nw_t"].abs() >= SIG_T, "metric"].tolist()
    cols = [f"{m}.R" for m in sig]
    corr = sum(g[cols].corr(method="spearman") for _, g in per) / per.ngroups
    corr.index = corr.columns = sig
    corr_long = corr.stack().rename("rho").rename_axis(["metric_a", "metric_b"]).reset_index()

    periods = sorted(set(period.astype(str)))
    sample = pd.DataFrame([{
        "sector": ", ".join(SECTORS), "start": periods[0], "end": periods[-1], "quarters": len(periods),
        "rows": len(d), "companies": d.index.get_level_values("Ticker").nunique(),
        "median_companies_per_quarter": per.size().median(), "buckets": prep.n_buckets, "target": y,
    }])

    out = Path(__file__).resolve().parent / f"{'_'.join(SECTORS).lower()}_factor_ic_{END_PERIOD}"
    out.mkdir(exist_ok=True)
    summary.round(4).to_csv(out / "ic_summary.csv", index=False)
    pd.DataFrame(quarterly).round(4).to_csv(out / "ic_quarterly.csv", index=False)
    corr_long.round(3).to_csv(out / "corr.csv", index=False)
    sample.to_csv(out / "sample.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary[["metric", "group", "direction", "mean_ic", "nw_t", "pct_pos", "spread_q1_q5"]]
          .round(3).to_string(index=False))
    print(f"\nWrote {out.relative_to(ROOT)}/ (ic_summary, ic_quarterly, corr, sample)")


if __name__ == "__main__":
    main()

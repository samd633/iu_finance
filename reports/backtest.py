"""
Annual-rebalance backtest of each sector's top-choice model (combo_ic.CHOSEN_BY_SECTOR).

Each year the in-index companies are ranked on their Q4 data and held from the end
of Q1 (31 March) for one year, the same window as the panel's Fwd_Return_1Yr for a
Q4 row. Portfolios, all total returns:
  - Top 10: the 10 best scores, equal weight.
  - Bottom 10: the 10 worst scores, equal weight.
  - Sector: every company in the year's universe, equal weight and market-cap weight.
  - S&P 500: SPY over the same window.

Scores are built as in combo_ic (mean Quality bucket + mean Growth bucket, quintiles,
1 = best). Quintile scores tie often, so ties are broken by the same model on
percentile ranks (PanelPrep with 100 buckets).

Universe: in-index companies whose Q4 report was filed by the start date (the panel
leaves the forward return blank otherwise). A blank return is checked against the
company's dividend-adjusted daily prices: if it traded at the start but stopped
before the end (acquired or delisted), it is held to its last close and then in
cash; if it traded through the end, its report wasn't filed in time and it is left
out of that year; if it has no price at the start, it is left out too. Every such
case is listed in backtest_missing.csv.

The chosen models were picked on data covering these years, so this is an
in-sample illustration of the model, not an out-of-sample test.

Run from the project root:
    python reports/backtest.py
Writes reports/<sector>_backtest_<last start year>/ for every sector in
CHOSEN_BY_SECTOR (git-ignored, like the other report CSVs).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from combo_ic import CHOSEN_BY_SECTOR  # noqa: E402
from fmp_call import _fetch_daily_prices  # noqa: E402
from panel_prep import PanelPrep  # noqa: E402

FIRST_PERIOD = "2011Q4"
N_HOLD = 10
ROLE = "Top choice"
RETURN = "Fwd_Return_1Yr.Y"
EXCESS = "Fwd_Excess_Return_1Yr.Y"
STALE_DAYS = 7          # a close more than this many days before a date doesn't count as on it
_prices: dict[str, pd.Series] = {}


def score(d: pd.DataFrame, model: str, n_buckets: int) -> pd.Series:
    """Model score per row: mean Quality bucket + mean Growth bucket (lower = better)."""
    def bucket(m: str) -> pd.Series:
        b = d[f"{m.lstrip('~')}.R"]
        if m.startswith("~"):  # flip actual ranks only, as combo_ic does
            b = b.where(d[f"{m.lstrip('~')}.F"].eq(1), n_buckets + 1 - b)
        return b
    quality, growth = ([m.strip() for m in side.split("+")] for side in model.split("|"))
    return (pd.concat([bucket(m) for m in quality], axis=1).mean(axis=1)
            + pd.concat([bucket(m) for m in growth], axis=1).mean(axis=1))


def close_on(px: pd.Series, date: pd.Timestamp) -> float | None:
    before = px[px.index <= date]
    return before.iloc[-1] if len(before) and (date - before.index[-1]).days <= STALE_DAYS else None


def resolve_missing(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, float | None]:
    """Why a forward return is blank, and the return to use (None = leave the company out)."""
    if ticker not in _prices:
        try:
            _prices[ticker] = _fetch_daily_prices(ticker, "2011-01-01", adjusted=True).dropna()
        except Exception:  # no price history at FMP for this ticker
            _prices[ticker] = pd.Series(dtype=float)
    px = _prices[ticker]
    p0 = close_on(px, start)
    if p0 is None:
        return "no price at start", None
    if close_on(px, end) is not None:
        return "not filed by start", None
    held = px[(px.index > start) & (px.index <= end)]
    if not len(held):
        return "no price after start", None
    return f"stopped trading {held.index[-1]:%Y-%m-%d}", (held.iloc[-1] / p0 - 1) * 100


def run_sector(sector: str, model: str):
    d5 = PanelPrep(sectors=[sector], target=None).prepare()
    d100 = PanelPrep(sectors=[sector], target=None, n_buckets=100).prepare()
    d5["score"], d5["score_pct"] = score(d5, model, 5), score(d100, model, 100).reindex(d5.index)
    periods = d5.index.get_level_values("Period")
    annual, holdings, missing = [], [], []
    for p in sorted(set(periods[(periods.quarter == 4) & (periods >= pd.Period(FIRST_PERIOD, "Q"))])):
        rows = d5.xs(p, level="Period").copy()
        spy = (rows[RETURN] - rows[EXCESS]).dropna()
        if spy.empty:  # the window hasn't ended yet
            continue
        start, end = (p + 1).end_time.normalize(), (p + 5).end_time.normalize()
        for t in rows.index[rows[RETURN].isna()]:
            reason, r = resolve_missing(t, start, end)
            rows.loc[t, RETURN] = r
            missing.append({"start": start.date(), "ticker": t, "reason": reason, "return_used": r})
        u = rows[rows[RETURN].notna()].sort_values(["score", "score_pct"])
        top, bottom = u.head(N_HOLD), u.tail(N_HOLD)[::-1]
        cap = u["Mkt_Cap.D"].where(u["Mkt_Cap.D"] > 0)
        annual.append({
            "window": f"{start:%b %Y} – {end:%b %Y}", "start": start.date(), "end": end.date(), "companies": len(u),
            "top10": top[RETURN].mean(), "bottom10": bottom[RETURN].mean(),
            "sector_ew": u[RETURN].mean(), "sector_cw": (u[RETURN] * cap).sum() / cap.sum(),
            "sp500": spy.median()})
        for side, g in (("Top 10", top), ("Bottom 10", bottom)):
            for i, (t, r) in enumerate(g.iterrows(), 1):
                holdings.append({"start": start.date(), "side": side, "position": i, "ticker": t,
                                 "industry_group": r["Industry_Group.D"], "score": r["score"], "return": r[RETURN]})
    annual = pd.DataFrame(annual)
    return annual, pd.DataFrame(holdings), pd.DataFrame(missing)


SERIES = {"top10": "Top 10", "bottom10": "Bottom 10", "sector_ew": "Sector (equal weight)",
          "sector_cw": "Sector (cap weight)", "sp500": "S&P 500"}


def summarize(annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    growth = [{"date": annual["start"].iloc[0], "series": SERIES[k], "value": 100.0} for k in SERIES]
    rows = []
    for k, name in SERIES.items():
        g = 100 * (1 + annual[k] / 100).cumprod()
        growth += [{"date": e, "series": name, "value": v} for e, v in zip(annual["end"], g)]
        peak = np.maximum.accumulate(np.r_[100.0, g.to_numpy()])
        rows.append({"series": name, "years": len(annual), "end_value": g.iloc[-1],
                     "cagr": ((g.iloc[-1] / 100) ** (1 / len(annual)) - 1) * 100,
                     "mean_annual": annual[k].mean(), "stdev_annual": annual[k].std(),
                     "best_year": annual[k].max(), "worst_year": annual[k].min(),
                     "max_drawdown": ((np.r_[100.0, g.to_numpy()] / peak).min() - 1) * 100,
                     "years_beat_sector_ew": int((annual[k] > annual["sector_ew"]).sum()) if k != "sector_ew" else None,
                     "years_beat_sp500": int((annual[k] > annual["sp500"]).sum()) if k != "sp500" else None})
    return pd.DataFrame(rows), pd.DataFrame(growth)


def main() -> None:
    for sector, chosen in CHOSEN_BY_SECTOR.items():
        annual, holdings, missing = run_sector(sector, chosen[ROLE])
        summary, growth = summarize(annual)
        slug = sector.lower().replace(" ", "_")
        dest = Path(__file__).resolve().parent / f"{slug}_backtest_{annual['start'].iloc[-1].year}"
        dest.mkdir(exist_ok=True)
        annual.round(3).to_csv(dest / "backtest_annual.csv", index=False)
        holdings.round(3).to_csv(dest / "backtest_holdings.csv", index=False)
        missing.round(3).to_csv(dest / "backtest_missing.csv", index=False)
        summary.round(3).to_csv(dest / "backtest_summary.csv", index=False)
        growth.round(3).to_csv(dest / "backtest_growth.csv", index=False)
        print(f"\n{sector}: {chosen[ROLE]} ({annual['window'].iloc[0]} to {annual['window'].iloc[-1]})")
        print(summary[["series", "end_value", "cagr", "stdev_annual", "worst_year", "max_drawdown",
                       "years_beat_sector_ew", "years_beat_sp500"]].round(1).to_string(index=False))
        print("missing returns:", missing["reason"].str.split().str[0].value_counts().to_dict() if len(missing) else {})


if __name__ == "__main__":
    main()

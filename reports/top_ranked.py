"""
Top-ranked companies under each sector's top-choice model (combo_ic.CHOSEN_BY_SECTOR)
in the latest complete quarter of the panel.

A quarter is complete when at least COMPLETE_SHARE of the previous quarter's
in-index companies have reported it; the newest quarter is usually still being
filed, and ranking a half-reported quarter would compare companies against a
partial sector. No forward return is needed, so this runs past combo_ic's
END_PERIOD.

Scores are built as in combo_ic: each metric's bucket (1 = best, missing values
filled as in PanelPrep.fill_missing, "~" metrics flipped), the mean of the
Quality buckets plus the mean of the Growth buckets; lower = better. Ties share
a rank, and every company tied at the cut-off is kept.

Run from the project root:
    python reports/top_ranked.py
Writes reports/<sector>_top_ranked_<period>/top_ranked.csv for every sector in
CHOSEN_BY_SECTOR (git-ignored, like the other report CSVs). Company names come
from the FMP profile endpoint (cached by fmp_call).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from combo_ic import CHOSEN_BY_SECTOR  # noqa: E402
from fmp_call import API_KEY, FMP_BASE, _get_json  # noqa: E402
from panel_prep import PanelPrep  # noqa: E402

TOP_N = 20
COMPLETE_SHARE = 0.9
ROLE = "Top choice"


def latest_complete(panel: pd.DataFrame) -> pd.Period:
    counts = panel[panel["In_Index.D"].eq("Y")].groupby(level="Period").size()
    complete = counts[counts >= COMPLETE_SHARE * counts.shift(1)]
    return complete.index[-1]


def company_name(ticker: str) -> str:
    records = _get_json(f"{FMP_BASE}/profile", {"symbol": ticker, "apikey": API_KEY})
    return records[0].get("companyName", "") if records else ""


def rank_sector(sector: str, model: str) -> pd.DataFrame:
    prep = PanelPrep(sectors=[sector], target=None)
    d = prep.prepare()
    period = latest_complete(d)
    d = d.xs(period, level="Period")
    quality, growth = ([m.strip() for m in side.split("+")] for side in model.split("|"))

    def bucket(m: str) -> pd.Series:
        b = d[f"{m.lstrip('~')}.R"]
        if m.startswith("~"):  # flip actual ranks only, as combo_ic does
            b = b.where(d[f"{m.lstrip('~')}.F"].eq(1), prep.n_buckets + 1 - b)
        return b

    out = pd.DataFrame({"industry_group": d["Industry_Group.D"]})
    for m in quality + growth:
        base = m.lstrip("~")
        out[f"{m} value"] = d[base]
        out[f"{m} bucket"] = bucket(m)
        out[f"{m} filled"] = d[f"{base}.F"].eq(1)
    out["score"] = (pd.concat([bucket(m) for m in quality], axis=1).mean(axis=1)
                    + pd.concat([bucket(m) for m in growth], axis=1).mean(axis=1))
    out["rank"] = out["score"].rank(method="min").astype(int)
    out = out[out["rank"] <= TOP_N].sort_values(["rank", "score"]).reset_index()
    out.insert(0, "period", str(period))
    out.insert(1, "sector", sector)
    out.insert(2, "model", model)
    out.insert(3, "companies", len(d))
    out.insert(5, "name", [company_name(t) for t in out["Ticker"]])
    return out.rename(columns={"Ticker": "ticker"})


def main() -> None:
    for sector, chosen in CHOSEN_BY_SECTOR.items():
        top = rank_sector(sector, chosen[ROLE])
        slug = sector.lower().replace(" ", "_")
        dest = Path(__file__).resolve().parent / f"{slug}_top_ranked_{top['period'].iloc[0]}"
        dest.mkdir(exist_ok=True)
        top.round(4).to_csv(dest / "top_ranked.csv", index=False)
        print(f"\n{sector} ({top['period'].iloc[0]}, {top['companies'].iloc[0]} companies): {chosen[ROLE]}")
        print(top[["rank", "ticker", "name", "industry_group", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""
S&P 500 sector accessor and sector-level analytics.

`get_sector_df()` reads the constituent records (ticker, company name, sector)
from SP500_Tick.JSON and makes no network calls -- the data is produced and
persisted by Scraper.SP500_Scrape() (see scraper.py).

`get_sector_weights()` builds on that by pulling each stock's current market
cap from FMP and computing its market-cap weight within a given sector. Those
lookups use the shared `fmp_cache`.
"""
import json
from pathlib import Path

import requests_cache
import numpy as np
import pandas as pd

from cagr import CAGR
from rolling_avg import RollingAvg
from eps_deviation import EPS_Deviation

FMP_BASE = "https://financialmodelingprep.com/stable"

with open("fmp_key.txt") as f:
    API_KEY = f.read().strip()


class SP_Sector:

    TICKER_FILE = "SP500_Tick.JSON"

    def __init__(self, ticker_file: str | None = None):
        self.ticker_file = ticker_file or self.TICKER_FILE
        self._fmp = requests_cache.CachedSession(
            "fmp_cache", expire_after=86400, allowable_codes=(200,)
        )

    # ------------------------------------------------------------------
    # Constituents (offline read)
    # ------------------------------------------------------------------

    def get_sector_df(self) -> pd.DataFrame:
        """One row per constituent: ticker, company_name, sector."""
        data = json.loads(Path(self.ticker_file).read_text())
        return pd.DataFrame(
            data["constituents"], columns=["ticker", "company_name", "sector"]
        )

    # ------------------------------------------------------------------
    # Market-cap weighting
    # ------------------------------------------------------------------

    def _fetch_market_cap(self, ticker: str) -> float | None:
        """Current market capitalisation from FMP. None if not returned."""
        resp = self._fmp.get(
            f"{FMP_BASE}/market-capitalization",
            params={"symbol": ticker, "apikey": API_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data[0].get("marketCap") if data else None

    def _fetch_latest_income(self, ticker: str) -> dict | None:
        """Most recent annual income statement from FMP. None if not returned."""
        resp = self._fmp.get(
            f"{FMP_BASE}/income-statement",
            params={"symbol": ticker, "period": "annual",
                    "limit": 1, "apikey": API_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data[0] if data else None

    def _build_sector_frame(self, sector: str) -> pd.DataFrame:
        """
        Full internal frame for a sector: per-company market cap, weight, the
        absolute income figures (net_income, revenue, shares) needed for sector
        aggregation, and the per-company per-share values. Sorted largest-first.
        """
        members = self.get_sector_df()
        members = members[members["sector"] == sector]

        df = members[["ticker", "company_name"]].copy()
        df["market_cap"] = df["ticker"].apply(self._fetch_market_cap)
        df = df.dropna(subset=["market_cap"])
        df["weight_pct"] = (df["market_cap"] / df["market_cap"].sum() * 100).round(2)

        income = df["ticker"].apply(self._fetch_latest_income)
        df["net_income"]  = income.apply(lambda r: r.get("netIncome") if r else None)
        df["revenue"]     = income.apply(lambda r: r.get("revenue") if r else None)
        df["shares"]      = income.apply(lambda r: r.get("weightedAverageShsOutDil") if r else None)
        df["eps_diluted"] = income.apply(lambda r: r.get("epsDiluted") if r else None)
        df["rev_per_share"] = (df["revenue"] / df["shares"]).round(2)

        return df.sort_values("market_cap", ascending=False).reset_index(drop=True)

    _DISPLAY_COLS = ["ticker", "company_name", "market_cap", "weight_pct",
                     "eps_diluted", "rev_per_share"]

    def get_sector_weights(self, sector: str = "Technology") -> pd.DataFrame:
        """
        Per-company view for `sector`:
            ticker, company_name, market_cap, weight_pct, eps_diluted, rev_per_share
        weight_pct is each stock's market cap as a percent of the sector total;
        the per-share figures are from each company's latest annual income
        statement. Sorted largest-to-smallest.
        """
        return self._build_sector_frame(sector)[self._DISPLAY_COLS]

    @staticmethod
    def _aggregate_summary(df: pd.DataFrame, sector: str) -> pd.DataFrame:
        """
        Sector treated as one consolidated entity:
            eps_diluted   = total net income / total diluted shares
            rev_per_share = total revenue    / total diluted shares
        """
        total_shares = df["shares"].sum()
        return pd.DataFrame([{
            "sector": sector,
            "market_cap": int(df["market_cap"].sum()),
            "eps_diluted": round(df["net_income"].sum() / total_shares, 2),
            "rev_per_share": round(df["revenue"].sum() / total_shares, 2),
        }])

    def get_sector_summary(self, sector: str = "Technology") -> pd.DataFrame:
        """Consolidated sector-level per-share metrics (one row)."""
        return self._aggregate_summary(self._build_sector_frame(sector), sector)

    # ------------------------------------------------------------------
    # Sector growth over time
    # ------------------------------------------------------------------

    def _fetch_income_history(self, ticker: str, n: int) -> list[dict]:
        """Last `n` annual income statements from FMP (newest first)."""
        resp = self._fmp.get(
            f"{FMP_BASE}/income-statement",
            params={"symbol": ticker, "period": "annual",
                    "limit": n, "apikey": API_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json() or []

    def _fetch_balance_history(self, ticker: str, n: int) -> list[dict]:
        """Last `n` annual balance sheets from FMP (newest first)."""
        resp = self._fmp.get(
            f"{FMP_BASE}/balance-sheet-statement",
            params={"symbol": ticker, "period": "annual",
                    "limit": n, "apikey": API_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json() or []

    def _fetch_cashflow_history(self, ticker: str, n: int) -> list[dict]:
        """Last `n` annual cash-flow statements from FMP (newest first)."""
        resp = self._fmp.get(
            f"{FMP_BASE}/cash-flow-statement",
            params={"symbol": ticker, "period": "annual",
                    "limit": n, "apikey": API_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json() or []

    def _company_year_components(self, ticker: str, n: int) -> list[dict]:
        """
        Per-fiscal-year raw components for one company, aligned across the
        income, balance-sheet and cash-flow statements. These are the absolute
        figures that get summed across the sector before any ratio is taken:
            net_income, revenue, shares, ebitda,
            ni_common  (net income available to common = net income - |pref divs|),
            avg_common_equity ((common_eq[t] + common_eq[t-1]) / 2)
        Balance sheet is pulled with one extra year so avg equity exists for the
        oldest income year. Common equity = total equity - preferred stock.
        """
        income  = {int(s["fiscalYear"]): s for s in self._fetch_income_history(ticker, n)
                   if s.get("fiscalYear")}
        cash    = {int(s["fiscalYear"]): s for s in self._fetch_cashflow_history(ticker, n)
                   if s.get("fiscalYear")}
        common_eq = {}
        for s in self._fetch_balance_history(ticker, n + 1):
            if not s.get("fiscalYear"):
                continue
            equity = s.get("totalStockholdersEquity")
            if equity is not None:
                common_eq[int(s["fiscalYear"])] = equity - (s.get("preferredStock") or 0)

        rows = []
        for fy, inc in income.items():
            cur, prev = common_eq.get(fy), common_eq.get(fy - 1)
            avg_ceq = (cur + prev) / 2 if (cur is not None and prev is not None) else None

            net_income = inc.get("netIncome")
            pref_div = (cash.get(fy) or {}).get("preferredDividendsPaid") or 0
            ni_common = net_income - abs(pref_div) if net_income is not None else None

            rows.append({
                "fiscal_year": fy,
                "ticker": ticker,
                "net_income": net_income,
                "revenue": inc.get("revenue"),
                "shares": inc.get("weightedAverageShsOutDil"),
                "ebitda": inc.get("ebitda"),
                "eps_diluted": inc.get("epsDiluted"),
                "ni_common": ni_common,
                "avg_common_equity": avg_ceq,
            })
        return rows

    def _build_sector_timeseries(self, sector: str, n: int,
                                 coverage: float = 0.90) -> pd.DataFrame:
        """
        Consolidated sector series, one row per fiscal year:
            fiscal_year, n_companies,
            eps_diluted, rev_per_share, ebitda_margin, roce
        Each metric is taken from sector-wide totals of the absolute components
        (Σ net income / Σ shares, Σ ebitda / Σ revenue, etc.), so the sector is
        treated as one consolidated entity. Ratio components are paired per
        company so a missing field never lands in only one side of a ratio.

        Only the current constituent list is used, aligned by FMP's `fiscalYear`
        label. Years are dropped unless their coverage reaches `coverage` x the
        peak coverage observed (1.0 = every current member reported; default
        0.90), which removes partially-reported recent years and thin early years.
        """
        tickers = self.get_sector_df()
        tickers = tickers[tickers["sector"] == sector]["ticker"]

        rows = []
        for ticker in tickers:
            rows.extend(self._company_year_components(ticker, n))

        long = pd.DataFrame(rows).dropna(subset=["net_income", "revenue", "shares"])

        # Pair ratio components so numerator and denominator share the same
        # company set within each year (NaN both out if either is missing).
        roce_ok = long["ni_common"].notna() & long["avg_common_equity"].notna()
        long["ni_common_p"] = long["ni_common"].where(roce_ok)
        long["avg_ceq_p"]   = long["avg_common_equity"].where(roce_ok)

        margin_ok = long["ebitda"].notna() & long["revenue"].notna()
        long["ebitda_p"]      = long["ebitda"].where(margin_ok)
        long["rev_margin_p"]  = long["revenue"].where(margin_ok)

        agg = (long.groupby("fiscal_year")
                   .agg(n_companies=("ticker", "nunique"),
                        net_income=("net_income", "sum"),
                        revenue=("revenue", "sum"),
                        shares=("shares", "sum"),
                        ni_common=("ni_common_p", "sum"),
                        avg_ceq=("avg_ceq_p", "sum"),
                        ebitda=("ebitda_p", "sum"),
                        rev_margin=("rev_margin_p", "sum"))
                   .reset_index()
                   .sort_values("fiscal_year")
                   .reset_index(drop=True))

        # Full-coverage filter: keep only years with a consistent member set.
        peak = agg["n_companies"].max()
        agg = agg[agg["n_companies"] >= coverage * peak].reset_index(drop=True)

        agg["eps_diluted"]   = (agg["net_income"] / agg["shares"]).round(2)
        agg["rev_per_share"] = (agg["revenue"] / agg["shares"]).round(2)
        agg["ebitda_margin"] = (agg["ebitda"] / agg["rev_margin"].replace(0, np.nan)).round(4)
        agg["roce"]          = (agg["ni_common"] / agg["avg_ceq"].replace(0, np.nan)).round(4)
        return agg

    def get_sector_growth(self, sector: str = "Technology", n: int = 10,
                          coverage: float = 0.90) -> pd.DataFrame:
        """
        Consolidated sector series with derived metrics:
            fiscal_year, n_companies,
            eps_diluted, rev_per_share, ebitda_margin, roce,
            eps_diluted_cagr_3y/5y, rev_per_share_cagr_3y/5y,
            ebitda_margin_avg_3y/5y, roce_avg_3y/5y,
            eps_diluted_std_dev, eps_diluted_downside_dev
        `coverage` controls the full-coverage year filter: a year is kept only
        if its reporting members reach `coverage` x the peak coverage observed
        (1.0 = strict; default 0.90).
        """
        ts = self._build_sector_timeseries(sector, n, coverage)
        ts = CAGR.compute(ts, ["eps_diluted", "rev_per_share"])
        ts = RollingAvg.compute(ts, ["ebitda_margin", "roce"])
        ts = EPS_Deviation.compute(ts, "eps_diluted")
        cols = ["fiscal_year", "n_companies",
                "eps_diluted", "rev_per_share", "ebitda_margin", "roce",
                "eps_diluted_cagr_3y", "eps_diluted_cagr_5y",
                "rev_per_share_cagr_3y", "rev_per_share_cagr_5y",
                "ebitda_margin_avg_3y", "ebitda_margin_avg_5y",
                "roce_avg_3y", "roce_avg_5y"
        ]
                # "eps_diluted_std_dev", "eps_diluted_downside_dev"]
        
        return ts[cols]

    def get_sector_year_detail(self, sector: str = "Technology", n: int = 10,
                               coverage: float = 0.90) -> tuple[int, pd.DataFrame]:
        """
        Per-company financials for the single most recent full-coverage year of
        the sector time series -- same year and same source data as
        get_sector_growth, so the stock detail reconciles with the aggregate.
        Returns (fiscal_year, DataFrame) where the frame holds:
            ticker, company_name, market_cap, weight_pct,
            eps_diluted, rev_per_share, ebitda_margin, roce
        market_cap / weight_pct are current (point-in-time); the financials are
        for the most recent full year. Sorted largest-to-smallest by market cap.
        """
        members = self.get_sector_df()
        members = members[members["sector"] == sector][["ticker", "company_name"]]

        rows = []
        for ticker in members["ticker"]:
            rows.extend(self._company_year_components(ticker, n))
        long = pd.DataFrame(rows).dropna(subset=["net_income", "revenue", "shares"])

        # Most recent year clearing the same coverage filter as the time series.
        counts = long.groupby("fiscal_year")["ticker"].nunique()
        valid = counts[counts >= coverage * counts.max()]
        year = int(valid.index.max())

        detail = long[long["fiscal_year"] == year].merge(members, on="ticker", how="left")
        detail["market_cap"] = detail["ticker"].apply(self._fetch_market_cap)
        detail["market_cap"] = round(detail["market_cap"] / 1_000_000, 2)
        detail["weight_pct"] = (detail["market_cap"] / detail["market_cap"].sum() * 100).round(2)
        detail["rev_per_share"] = (detail["revenue"] / detail["shares"]).round(2)
        detail["ebitda_margin"] = (detail["ebitda"] / detail["revenue"].replace(0, np.nan)).round(4)
        detail["roce"]          = (detail["ni_common"] / detail["avg_common_equity"].replace(0, np.nan)).round(4)

        cols = ["ticker", "company_name", "market_cap", "weight_pct",
                "eps_diluted", "rev_per_share", "ebitda_margin", "roce"]
        detail = detail.sort_values("market_cap", ascending=False)[cols].reset_index(drop=True)
        detail = detail.rename(columns={'market_cap': 'market_cap (MM)'})
        return year, detail

    # ------------------------------------------------------------------

    def main(self, sector: str = "Healthcare"):
        print(f"{sector} — Sector-Level Historic Metrics")
        print("=" * 70)
        print(self.get_sector_growth(sector).to_string(index=False))

        year, detail = self.get_sector_year_detail(sector)
        print()
        print(f"{sector} — Stock-Level Detail (FY{year})")
        print("=" * 70)
        print(detail.to_string(index=False))


if __name__ == "__main__":
    SP_Sector().main()


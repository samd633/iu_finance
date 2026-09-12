"""
FMP API data-fetching layer. Returns raw DataFrames — no derived metrics.
"""
import requests
import requests_cache
import pandas as pd

FMP_BASE = "https://financialmodelingprep.com/stable"

requests_cache.install_cache('fmp_cache', expire_after=86400, allowable_codes=(200,))

with open("fmp_key.txt") as f:
    API_KEY = f.read().strip()


class FMP_Financial:
    def __init__(self, ticker: str, n: int = 10):
        self.ticker = ticker.upper()
        self.n = n
        self._income_stmt   = None
        self._balance_sheet = None
        self._cash_flow     = None
        self._prices        = None
        self._dividends     = None

    # ------------------------------------------------------------------
    # Private fetchers
    # ------------------------------------------------------------------

    def _fetch(self, endpoint: str, **extra_params) -> list[dict]:
        resp = requests.get(
            f"{FMP_BASE}/{endpoint}",
            params={"symbol": self.ticker, "period": "annual",
                    "apikey": API_KEY, **extra_params},
        )
        resp.raise_for_status()
        return resp.json()

    def _get_income_stmt(self) -> list[dict]:
        if self._income_stmt is None:
            self._income_stmt = self._fetch("income-statement", limit=self.n)
        return self._income_stmt

    def _get_balance_sheet(self) -> list[dict]:
        if self._balance_sheet is None:
            # One extra year so callers can compute prior-year averages for all n years.
            self._balance_sheet = self._fetch("balance-sheet-statement", limit=self.n + 1)
        return self._balance_sheet

    def _get_cash_flow(self) -> list[dict]:
        if self._cash_flow is None:
            self._cash_flow = self._fetch("cash-flow-statement", limit=self.n)
        return self._cash_flow

    def _get_prices(self) -> pd.Series:
        if self._prices is None:
            dates = sorted(item["date"] for item in self._get_income_stmt())
            # Start 7 days before the first fiscal year-end to cover weekend/holiday dates.
            start = (pd.Timestamp(dates[0]) - pd.DateOffset(days=7)).strftime("%Y-%m-%d")
            resp = requests.get(
                f"{FMP_BASE}/historical-price-eod/full",
                params={"symbol": self.ticker, "from": start,
                        "to": dates[-1], "apikey": API_KEY},
            )
            resp.raise_for_status()
            records = resp.json()
            self._prices = pd.Series(
                {pd.Timestamp(r["date"]): r["close"] for r in records}
            ).sort_index()
        return self._prices

    # ------------------------------------------------------------------
    # Public data accessors
    # ------------------------------------------------------------------

    def get_income_stmt(self) -> pd.DataFrame:
        df = pd.DataFrame(self._get_income_stmt())[
            ["date", "fiscalYear", "epsDiluted", "revenue",
             "weightedAverageShsOutDil", "ebitda", "netIncome"]
        ].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def get_balance_sheet(self) -> pd.DataFrame:
        df = pd.DataFrame(self._get_balance_sheet())[
            ["date", "totalStockholdersEquity", "preferredStock"]
        ].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def get_cash_flow(self) -> pd.DataFrame:
        df = pd.DataFrame(self._get_cash_flow())[
            ["date", "operatingCashFlow", "preferredDividendsPaid"]
        ].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def get_prices(self) -> pd.Series:
        """Daily closing prices as a date-indexed Series, sorted ascending."""
        return self._get_prices()

    def get_dividends(self) -> pd.Series:
        """Adjusted dividends as an ex-dividend-date-indexed Series, sorted ascending."""
        if self._dividends is None:
            resp = requests.get(
                f"{FMP_BASE}/dividends",
                params={"symbol": self.ticker, "apikey": API_KEY},
            )
            resp.raise_for_status()
            self._dividends = resp.json()
        records = self._dividends
        return pd.Series(
            {pd.Timestamp(r["date"]): r["adjDividend"] for r in records}
        ).sort_index()

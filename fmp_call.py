"""
FMP API data-fetching layer. Returns raw DataFrames — no derived metrics.
"""
import re
import json
import requests
import requests_cache
import pandas as pd

FMP_BASE = "https://financialmodelingprep.com/stable"
requests_cache.install_cache('fmp_cache', expire_after=86400, allowable_codes=(200,))

with open("fmp_key.txt") as f:
    API_KEY = f.read().strip()

with open("metrics.JSON") as f:
    METRICS_CFG = json.load(f)


_FORMULA_FUNCS = {
    # Row-wise (per report date) sample standard deviation across N shifted values,
    # e.g. STDV(EPS.0, EPS.1, EPS.2, EPS.3, EPS.4) for a trailing 5-period variance.
    "STDV": lambda *series: pd.concat(series, axis=1).std(axis=1),
}


def _eval_formula(formula: str, df: pd.DataFrame) -> pd.Series:
    """Evaluate a metrics.JSON formula against a DataFrame of named field columns.

    FIELD.N syntax shifts the column N periods backward (e.g. COMM.1 = prior year COMM).
    """
    field_names = set(df.columns)

    def replacer(m):
        name = m.group(1)
        if name not in field_names:
            return m.group(0)
        shift = int(m.group(2)) if m.group(2) is not None else 0
        if shift == 0:
            return f'__df["{name}"]'
        return f'__df["{name}"].shift({shift})'

    transformed = re.sub(r'\b([A-Za-z_][A-Za-z0-9_]*)(?:\.(\d+))?', replacer, formula)
    return eval(transformed, {"__df": df, **_FORMULA_FUNCS})


class FMP_Financial:
    def __init__(self,
                 ticker: str,
                 period: str = "annual",
                 n: int = 10):
        self.ticker = ticker.upper()
        self.period = period
        self.n = n
        self._income_stmt   = None
        self._balance_sheet = None
        self._cash_flow     = None
        self._daily_prices  = None
        self._prices        = None
        self._lookback_prices = {}

    def _fetch(self, endpoint: str, **extra_params) -> list[dict]:
        resp = requests.get(
            f"{FMP_BASE}/{endpoint}",
            params={"symbol": self.ticker, "period": self.period,
                    "apikey": API_KEY, **extra_params},
        )
        resp.raise_for_status()
        return resp.json()

    def _get_income_stmt(self) -> list[dict]:
        if self._income_stmt is None:
            self._income_stmt = self._fetch("income-statement", limit=self.n + 5)
        return self._income_stmt

    def _get_balance_sheet(self) -> list[dict]:
        if self._balance_sheet is None:
            # One extra year so prior-year shift formulas (e.g. AVG_COMM) cover all n IS dates.
            self._balance_sheet = self._fetch("balance-sheet-statement", limit=self.n + 5 + 1)
        return self._balance_sheet

    def _get_cash_flow(self) -> list[dict]:
        if self._cash_flow is None:
            self._cash_flow = self._fetch("cash-flow-statement", limit=self.n + 5)
        return self._cash_flow

    def _get_filings(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """(period-end date, SEC filing date) pairs, one per income-statement period."""
        return sorted(
            (pd.Timestamp(item["date"]), pd.Timestamp(item["filingDate"]))
            for item in self._get_income_stmt()
        )

    def _get_daily_prices(self) -> pd.Series:
        """Daily close prices spanning from 12 months before the earliest filing
        date through the 3-day post-filing window after the latest — far enough
        back to cover the 12-month momentum lookback for every period.
        """
        if self._daily_prices is None:
            filings = self._get_filings()
            start = (filings[0][1] - pd.DateOffset(months=12, days=10)).strftime("%Y-%m-%d")
            end = (filings[-1][1] + pd.DateOffset(days=10)).strftime("%Y-%m-%d")
            resp = requests.get(
                f"{FMP_BASE}/historical-price-eod/full",
                params={"symbol": self.ticker, "from": start,
                        "to": end, "apikey": API_KEY},
            )
            resp.raise_for_status()
            records = resp.json()
            self._daily_prices = pd.Series(
                {pd.Timestamp(r["date"]): r["close"] for r in records}
            ).sort_index()
        return self._daily_prices

    def _get_prices(self) -> pd.Series:
        """Average close price over the 3 trading days on/after each period's SEC
        filing date (not the fiscal period-end date), indexed by period-end date.

        The filing date is when financials actually become public, so pricing off
        it avoids look-ahead bias; averaging a few days past it smooths the
        immediate earnings-reaction jump rather than pricing off a single day.
        """
        if self._prices is None:
            daily = self._get_daily_prices()
            self._prices = pd.Series({
                period_end: daily[daily.index >= filing_date].iloc[:3].mean()
                for period_end, filing_date in self._get_filings()
            }).sort_index()
        return self._prices

    def _get_lookback_prices(self, months: int) -> pd.Series:
        """Close price ~`months` calendar months before each period's filing date
        (the closest trading day on or before that point), indexed by period-end
        date — the same filing-date anchor the Price field is measured from, so
        momentum returns and P/E pricing share a consistent end date.
        """
        if months not in self._lookback_prices:
            daily = self._get_daily_prices()
            result = {}
            for period_end, filing_date in self._get_filings():
                target = filing_date - pd.DateOffset(months=months)
                window = daily[daily.index <= target]
                result[period_end] = window.iloc[-1] if not window.empty else None
            self._lookback_prices[months] = pd.Series(result).sort_index()
        return self._lookback_prices[months]

    def _get_fields(self) -> pd.DataFrame:
        fields_cfg = METRICS_CFG["Fields"]

        bs_raw = pd.DataFrame(self._get_balance_sheet())
        is_raw = pd.DataFrame(self._get_income_stmt())
        cf_raw = pd.DataFrame(self._get_cash_flow())

        for raw in [bs_raw, is_raw, cf_raw]:
            raw["date"] = pd.to_datetime(raw["date"])

        bs  = bs_raw.set_index("date").sort_index()
        is_ = is_raw.set_index("date").sort_index()
        cf  = cf_raw.set_index("date").sort_index()

        source_map = {"BS": bs, "IS": is_, "CF": cf}

        # Index on BS (n+1 rows) so shift-based formula fields resolve for all n IS dates.
        result = pd.DataFrame(index=bs.index)

        for field_name, field_cfg in fields_cfg.items():
            source = field_cfg["Source"]
            formula = field_cfg.get("Formula")
            rounding = field_cfg.get("Rounding")

            if source in source_map:
                result[field_name] = source_map[source][field_cfg["API"]].reindex(result.index)
                if formula and formula != "None":
                    # Partial expression (e.g. "/ 1_000_000") — prepend field name to form a full expr.
                    result[field_name] = _eval_formula(f"{field_name} {formula}", result)
            elif source == "Formula":
                result[field_name] = _eval_formula(formula, result)
            elif source == "Price":
                result[field_name] = self._get_prices().reindex(result.index)
            elif source == "PriceLookback":
                result[field_name] = self._get_lookback_prices(int(field_cfg["API"])).reindex(result.index)

            if rounding and rounding != "None":
                result[field_name] = result[field_name].round(int(rounding))

        # Trim to IS dates — drops the extra oldest BS row used for prior-year shifts.
        return result.reindex(is_.index)

    def _get_metrics(self) -> pd.DataFrame:
        fields = self._get_fields()
        result = pd.DataFrame(index=fields.index)

        for subset in METRICS_CFG["Metrics"].values():
            for metric_name, metric_cfg in subset.items():
                rounding = metric_cfg.get("Rounding")
                result[metric_name] = _eval_formula(metric_cfg["Formula"], fields)
                if rounding and rounding != "None":
                    result[metric_name] = result[metric_name].round(int(rounding))

        return result.iloc[-self.n:]


if __name__ == "__main__":
    print(FMP_Financial("NVDA")._get_metrics())

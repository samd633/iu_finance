"""
FMP API data-fetching + TTM calendarization layer. Returns raw-derived
DataFrames — no cross-cutting judgment calls beyond calendarization itself.

Annual-equivalent metrics are built from quarterly statements as
trailing-twelve-month (TTM) aggregates, one per calendar year, snapped to the
quarter-end nearest a target calendar date (default Dec 31). This puts
companies with different fiscal year-ends (AAPL ~Sep, MSFT Jun 30, NVDA
~Jan, ...) on the same economic period so cross-sectional comparisons are
apples-to-apples:

  - Flow fields (income statement, cash flow) are TTM sums of the 4 quarters
    ending nearest the target date (period averages, e.g. diluted share
    count, are TTM-averaged instead of summed).
  - Stock fields (balance sheet) are the point-in-time snapshot from that
    same quarter-end.
  - When no quarter actually lands on the target date, we snap to the
    nearest reported quarter-end rather than interpolating; the actual date
    used and its offset from the target are reported alongside each row
    (`_actual_period_end`, `_offset_days`) so a large snap (e.g. NVDA vs a
    Dec 31 target) is visible rather than hidden.
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

# APIs that represent a period average/rate rather than a period flow —
# TTM-aggregate with a rolling mean instead of a rolling sum.
_AVERAGE_APIS = {"weightedAverageShsOutDil"}


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
                 n: int = 10,
                 month: int = 12,
                 day: int = 31,
                 max_offset_days: int = 45):
        """
        n: number of calendarized annual (TTM) periods to return.
        month, day: target calendar date each TTM period is snapped to
            (default Dec 31), so different tickers can be compared on the
            same reporting date regardless of fiscal year-end.
        max_offset_days: a snapped period is dropped if its actual quarter-end
            is more than this many days from the target date (default 45,
            half a quarter) — guards against a not-yet-reported current
            period snapping to a stale prior quarter and masquerading as a
            full year of data.
        """
        self.ticker = ticker.upper()
        self.period = "quarter"  # all fetches are quarterly; annual-equivalent rows are TTM, calendarized
        self.n = n
        self.month = month
        self.day = day
        self.max_offset_days = max_offset_days
        # Quarters of raw history to pull: n requested years + 5 years for
        # deep-shift metrics (5Yr CAGR/variance) + 1 year for the AVG_*
        # field's own prior-year shift + 2 years slack for the TTM rolling
        # window and quarter-end snapping at the earliest selected point.
        self._n_quarters = (n + 8) * 4
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
            self._income_stmt = self._fetch("income-statement", limit=self._n_quarters)
        return self._income_stmt

    def _get_balance_sheet(self) -> list[dict]:
        if self._balance_sheet is None:
            # One extra year of quarters so prior-year shift formulas (e.g. AVG_COMM)
            # cover every calendarized date we might select.
            self._balance_sheet = self._fetch("balance-sheet-statement", limit=self._n_quarters + 4)
        return self._balance_sheet

    def _get_cash_flow(self) -> list[dict]:
        if self._cash_flow is None:
            self._cash_flow = self._fetch("cash-flow-statement", limit=self._n_quarters)
        return self._cash_flow

    def _get_filings(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """(period-end date, SEC filing date) pairs, one per quarterly income-statement period."""
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

    def _select_dates(self, available: pd.DatetimeIndex, num_years: int) -> dict:
        """{target_calendar_year: nearest available quarter-end date}, for
        num_years+1 consecutive calendar years ending at the latest year with data.
        """
        available = available.sort_values()
        max_year = available.max().year
        chosen = {}
        for y in range(max_year - num_years, max_year + 1):
            target = pd.Timestamp(year=y, month=self.month, day=self.day)
            diffs = available.map(lambda d: abs((d - target).days))
            chosen[y] = available[diffs.argmin()]
        return chosen

    def _get_primitive_fields(self) -> pd.DataFrame:
        """BS/IS/CF-sourced fields only, at quarterly frequency: flow fields
        are TTM (trailing-4-quarter) aggregates, stock fields are snapshots.
        Formula/Price/PriceLookback fields are deferred to annual frequency,
        where FIELD.N shifts mean "N calendar years back".
        """
        fields_cfg = METRICS_CFG["Fields"]

        bs_raw = pd.DataFrame(self._get_balance_sheet())
        is_raw = pd.DataFrame(self._get_income_stmt())
        cf_raw = pd.DataFrame(self._get_cash_flow())
        for raw in (bs_raw, is_raw, cf_raw):
            raw["date"] = pd.to_datetime(raw["date"])

        bs = bs_raw.set_index("date").sort_index()
        is_ = is_raw.set_index("date").sort_index()
        cf = cf_raw.set_index("date").sort_index()
        flow_map = {"IS": is_, "CF": cf}

        result = pd.DataFrame(index=bs.index)
        for field_name, field_cfg in fields_cfg.items():
            source = field_cfg["Source"]
            if source == "BS":
                col = bs[field_cfg["API"]].reindex(result.index)
            elif source in flow_map:
                api = field_cfg["API"]
                raw_col = flow_map[source][api]
                agg = (raw_col.rolling(4, min_periods=4).mean()
                       if api in _AVERAGE_APIS else
                       raw_col.rolling(4, min_periods=4).sum())
                col = agg.reindex(result.index)
            else:
                continue  # Formula / Price / PriceLookback — handled in _get_fields

            result[field_name] = col
            formula = field_cfg.get("Formula")
            if formula and formula != "None":
                # Partial expression (e.g. "/ 1_000_000") — prepend field name to form a full expr.
                result[field_name] = _eval_formula(f"{field_name} {formula}", result)
            rounding = field_cfg.get("Rounding")
            if rounding and rounding != "None":
                result[field_name] = result[field_name].round(int(rounding))

        return result

    def _get_fields(self) -> pd.DataFrame:
        fields_cfg = METRICS_CFG["Fields"]
        primitives = self._get_primitive_fields()

        # n requested + 5 years for deep-shift metrics + 1 year for AVG_* fields'
        # own prior-year shift = n+6 rows needed by _get_metrics before its final trim.
        chosen = self._select_dates(primitives.index, self.n + 6)
        years_sorted = sorted(chosen)
        dates = [chosen[y] for y in years_sorted]

        annual = primitives.loc[dates].copy()
        annual.index = years_sorted
        annual["_actual_period_end"] = dates
        annual["_offset_days"] = [
            (d - pd.Timestamp(year=y, month=self.month, day=self.day)).days
            for y, d in zip(years_sorted, dates)
        ]

        # Price / PriceLookback fields, resolved at the chosen dates.
        for field_name, field_cfg in fields_cfg.items():
            source = field_cfg["Source"]
            if source == "Price":
                s = self._get_prices().reindex(dates)
            elif source == "PriceLookback":
                s = self._get_lookback_prices(int(field_cfg["API"])).reindex(dates)
            else:
                continue
            s.index = years_sorted
            annual[field_name] = s
            rounding = field_cfg.get("Rounding")
            if rounding and rounding != "None":
                annual[field_name] = annual[field_name].round(int(rounding))

        # Formula fields, in metrics.JSON declaration order so dependencies
        # (e.g. ROE needs AVG_COMM needs COMM) resolve correctly.
        for field_name, field_cfg in fields_cfg.items():
            if field_cfg["Source"] != "Formula":
                continue
            annual[field_name] = _eval_formula(field_cfg["Formula"], annual)
            rounding = field_cfg.get("Rounding")
            if rounding and rounding != "None":
                annual[field_name] = annual[field_name].round(int(rounding))

        # Drop periods that snapped too far from the target date — most
        # commonly the current, still-in-progress year before its
        # near-target quarter has been reported.
        return annual[annual["_offset_days"].abs() <= self.max_offset_days]

    def _get_metrics(self) -> pd.DataFrame:
        fields = self._get_fields()
        result = pd.DataFrame(index=fields.index)

        for subset in METRICS_CFG["Metrics"].values():
            for metric_name, metric_cfg in subset.items():
                rounding = metric_cfg.get("Rounding")
                result[metric_name] = _eval_formula(metric_cfg["Formula"], fields)
                if rounding and rounding != "None":
                    result[metric_name] = result[metric_name].round(int(rounding))

        result["_actual_period_end"] = fields["_actual_period_end"]
        result["_offset_days"] = fields["_offset_days"]
        return result.iloc[-self.n:]


if __name__ == "__main__":
    FMP_Financial("MSFT")._get_metrics().to_csv("msft_metrics.csv", index_label="Year")

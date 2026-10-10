"""
FMP API data-fetching + TTM calendarization layer. Returns raw-derived
DataFrames — no cross-cutting judgment calls beyond calendarization itself.

Metrics are built from quarterly statements as trailing-twelve-month (TTM)
aggregates, one row per calendar period, each snapped to the nearest
calendar target date. With freq="Q" (default) the targets are calendar
quarter-ends (Mar 31, Jun 30, Sep 30, Dec 31); with freq="A" there is one
target per year (default Dec 31). This puts companies with different fiscal
year-ends (AAPL ~Sep, MSFT Jun 30, NVDA ~Jan, ...) on the same economic
period so cross-sectional comparisons are apples-to-apples:

  - Flow fields (income statement, cash flow) are TTM sums of the 4 quarters
    ending at the snapped quarter-end (period averages, e.g. diluted share
    count, are TTM-averaged instead of summed). Quarterly rows are still TTM,
    so ratios like P/E stay annualized and free of seasonality.
  - Stock fields (balance sheet) are the point-in-time snapshot from that
    same quarter-end.
  - When no quarter actually lands on the target date, we snap to the
    nearest reported quarter-end rather than interpolating; the actual date
    used and its offset from the target are reported alongside each row
    (`_actual_period_end`, `_offset_days`) so a large snap (e.g. NVDA vs a
    calendar quarter-end) is visible rather than hidden.
  - FIELD.N in metrics.JSON always means "N years back", whatever the
    frequency (it shifts 4N rows on quarterly data), so a formula like
    EPS_CAGR_5Yr means the same thing at either frequency.
"""
import re
import json
import time
import threading
import requests
import requests_cache
import pandas as pd

from sp_membership import SP500Membership

FMP_BASE = "https://financialmodelingprep.com/stable"
requests_cache.install_cache('fmp_cache', expire_after=86400, allowable_codes=(200,))

with open("fmp_key.txt") as f:
    API_KEY = f.read().strip()

with open("metrics.JSON") as f:
    METRICS_CFG = json.load(f)

# APIs that represent a period average/rate rather than a period flow —
# TTM-aggregate with a rolling mean instead of a rolling sum.
_AVERAGE_APIS = {"weightedAverageShsOutDil"}

# Sector/industry for former S&P 500 members whose FMP profile is missing
# (delisted, no profile) or now describes a different company that reused
# the ticker. Values use FMP's own taxonomy so they group with live profiles.
PROFILE_OVERRIDES = {
    "AABA": ("Communication Services", "Internet Content & Information"),  # Yahoo while in the index (left 2017-06)
    "AGN":  ("Healthcare", "Drug Manufacturers - Specialty & Generic"),     # Allergan
    "APC":  ("Energy", "Oil & Gas Exploration & Production"),               # Anadarko; profile is now ARKO Petroleum
    "CELG": ("Healthcare", "Biotechnology"),                                # Celgene
    "CXO":  ("Energy", "Oil & Gas Exploration & Production"),               # Concho Resources
    "ETFC": ("Financial Services", "Financial - Capital Markets"),          # E*TRADE
    "FLIR": ("Technology", "Hardware, Equipment & Parts"),                  # FLIR Systems (as acquirer Teledyne)
    "FTR":  ("Communication Services", "Telecommunications Services"),      # Frontier Communications
    "NBL":  ("Energy", "Oil & Gas Exploration & Production"),               # Noble Energy
    "RHT":  ("Technology", "Software - Infrastructure"),                    # Red Hat
    "RTN":  ("Industrials", "Aerospace & Defense"),                         # Raytheon
    "SNI":  ("Communication Services", "Entertainment"),                    # Scripps Networks
    "SPLS": ("Consumer Cyclical", "Specialty Retail"),                      # Staples; profile is now a PIMCO ETF
    "TIF":  ("Consumer Cyclical", "Luxury Goods"),                          # Tiffany
    "TSS":  ("Financial Services", "Financial - Credit Services"),          # Total System Services (as acquirer GPN)
    "VAR":  ("Healthcare", "Medical - Instruments & Supplies"),             # Varian Medical
    "VIAB": ("Communication Services", "Entertainment"),                    # Viacom
    "XEC":  ("Energy", "Oil & Gas Exploration & Production"),               # Cimarex Energy
}

# Max days between a balance-sheet / cash-flow record and the income-statement
# quarter-end it is matched to (see _get_primitive_fields).
ALIGN_TOLERANCE_DAYS = 10

_MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Shared across instances: building it downloads the full membership history.
_MEMBERSHIP = None
_MEMBERSHIP_LOCK = threading.Lock()

# Benchmark (non-ticker) daily prices, e.g. ^GSPC / SPY, shared across
# instances so a multi-ticker run fetches each benchmark once rather than
# once per ticker. Fetched from a fixed early date to cover any ticker.
_BENCHMARK_START = "1995-01-01"
_BENCHMARK_PRICES: dict[tuple[str, bool], pd.Series] = {}
_BENCHMARK_LOCK = threading.Lock()


def _get_membership() -> SP500Membership:
    global _MEMBERSHIP
    with _MEMBERSHIP_LOCK:
        if _MEMBERSHIP is None:
            _MEMBERSHIP = SP500Membership(API_KEY)
    return _MEMBERSHIP


def _get_json(url: str, params: dict):
    """GET with retry/backoff on HTTP 429 (FMP per-minute rate limit)."""
    for attempt in range(6):
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code != 429:
            break
        time.sleep(min(60, 5 * 2 ** attempt))
    resp.raise_for_status()
    return resp.json()


def _dedupe_statements(records: list[dict]) -> list[dict]:
    """One statement per period-end date.

    FMP occasionally returns the same quarter twice under different fiscal
    labels (e.g. AIV's 2024-12-31 balance sheet as both FY2024 Q4 and FY2025
    Q3, with different values). Keep the record whose (period, fiscal-year
    offset) matches the company's usual label for that calendar month, as
    learned from its non-duplicated quarters; fall back to the latest-accepted.
    """
    df = pd.DataFrame(records)
    if df.empty or not df["date"].duplicated().any():
        return records
    month = pd.to_datetime(df["date"]).dt.month
    label = list(zip(df["period"], df["fiscalYear"].astype(int) - pd.to_datetime(df["date"]).dt.year))
    unique = ~df["date"].duplicated(keep=False)
    usual = (pd.DataFrame({"month": month[unique], "label": [label[i] for i in df.index[unique]]})
             .groupby("month")["label"].agg(lambda s: s.mode().iloc[0]))
    df["_fits"] = [usual.get(m) == lab for m, lab in zip(month, label)]
    df["_accepted"] = df.get("acceptedDate", df.get("filingDate"))
    keep = (df.sort_values(["_fits", "_accepted"])
              .drop_duplicates("date", keep="last").index)
    return [records[i] for i in sorted(keep)]


def _fetch_daily_bars(symbol: str, start: str, adjusted: bool) -> pd.DataFrame:
    """Daily close and volume from `start` through today, fetched in 10-year
    chunks because FMP truncates each response to 5,000 rows (~20 years)."""
    endpoint, close = ("dividend-adjusted", "adjClose") if adjusted else ("full", "close")
    today = pd.Timestamp.today().normalize()
    bars = {}
    chunk_start = pd.Timestamp(start)
    while chunk_start <= today:
        chunk_end = min(chunk_start + pd.DateOffset(years=10, days=-1), today)
        records = _get_json(
            f"{FMP_BASE}/historical-price-eod/{endpoint}",
            {"symbol": symbol, "from": chunk_start.strftime("%Y-%m-%d"),
             "to": chunk_end.strftime("%Y-%m-%d"), "apikey": API_KEY},
        )
        bars.update({pd.Timestamp(r["date"]): (r[close], r.get("volume")) for r in records})
        chunk_start = chunk_end + pd.Timedelta(days=1)
    return pd.DataFrame.from_dict(bars, orient="index", columns=["close", "volume"],
                                  dtype=float).sort_index()


def _fetch_daily_prices(symbol: str, start: str, adjusted: bool) -> pd.Series:
    """Daily closes from `start` through today (see _fetch_daily_bars)."""
    return _fetch_daily_bars(symbol, start, adjusted)["close"]


_FORMULA_FUNCS = {
    # Row-wise (per report date) sample standard deviation across N shifted values,
    # e.g. STDV(EPS.0, EPS.1, EPS.2, EPS.3, EPS.4) for a trailing 5-period standard deviation.
    "STDV": lambda *series: pd.concat(series, axis=1).std(axis=1),
    # Row-wise minimum across N shifted values; NaN if any value is missing.
    "MIN": lambda *series: pd.concat(series, axis=1).min(axis=1, skipna=False),
}


def _eval_formula(formula: str, df: pd.DataFrame, periods_per_year: int = 1) -> pd.Series:
    """Evaluate a metrics.JSON formula against a DataFrame of named field columns.

    FIELD.N syntax shifts the column N *years* backward (e.g. COMM.1 = prior
    year COMM), i.e. N * periods_per_year rows. ROLL_MEAN(FIELD, N) averages
    every period in the trailing N years, current included (20 quarters for
    N=5 on quarterly data); NaN if any period in the window is missing.
    `df` must have one row per consecutive period (gaps filled with NaN rows)
    for shifts and windows to be calendar-exact.
    """
    field_names = set(df.columns)
    funcs = {
        **_FORMULA_FUNCS,
        "ROLL_MEAN": lambda s, years: s.rolling(years * periods_per_year,
                                                min_periods=years * periods_per_year).mean(),
    }

    def replacer(m):
        name = m.group(1)
        if name not in field_names:
            return m.group(0)
        shift = int(m.group(2)) * periods_per_year if m.group(2) is not None else 0
        if shift == 0:
            return f'__df["{name}"]'
        return f'__df["{name}"].shift({shift})'

    transformed = re.sub(r'\b([A-Za-z_][A-Za-z0-9_]*)(?:\.(\d+))?', replacer, formula)
    return eval(transformed, {"__df": df, **funcs})


class FMP_Financial:
    def __init__(self,
                 ticker: str,
                 n: int = 40,
                 freq: str = "Q",
                 month: int = 12,
                 day: int = 31,
                 max_offset_days: int = 45):
        """
        n: number of calendarized (TTM) periods to return — quarters when
            freq="Q" (default 40 = 10 years), years when freq="A".
        freq: "Q" for one row per calendar quarter-end, "A" for one row per
            year at the month/day target.
        month, day: annual target date (freq="A" only, default Dec 31), so
            different tickers can be compared on the same reporting date
            regardless of fiscal year-end.
        max_offset_days: a reported quarter-end is dropped if it is more than
            this many days from its nearest target date (default 45, half a
            quarter) — on annual data this keeps off-cycle quarters (and a
            not-yet-reported current year) from masquerading as a year-end.
        """
        if freq not in ("Q", "A"):
            raise ValueError(f"freq must be 'Q' or 'A', got {freq!r}")
        self.ticker = ticker.upper()
        self.period = "quarter"  # all fetches are quarterly; output rows are TTM, calendarized
        self.n = n
        self.freq = freq
        self.month = month
        self.day = day
        self.max_offset_days = max_offset_days
        self._periods_per_year = 4 if freq == "Q" else 1
        self._period_freq = "Q-DEC" if freq == "Q" else f"A-{_MONTH_ABBR[month - 1]}"
        # Quarters of raw history to pull: n requested periods + 5 years for
        # deep-shift metrics (5Yr CAGR/variance) + 1 year for the AVG_*
        # field's own prior-year shift + 2 years slack for the TTM rolling
        # window and quarter-end snapping at the earliest selected point.
        self._n_quarters = n * (4 // self._periods_per_year) + 8 * 4
        self._income_stmt   = None
        self._balance_sheet = None
        self._cash_flow     = None
        self._profile       = None
        self._daily_prices  = {}
        self._prices        = {}
        self._offset_prices = {}

    def _fetch(self, endpoint: str, **extra_params) -> list[dict]:
        records = _get_json(
            f"{FMP_BASE}/{endpoint}",
            {"symbol": self.ticker, "period": self.period,
             "apikey": API_KEY, **extra_params},
        )
        if not records:
            raise ValueError(f"FMP returned no {endpoint} data for {self.ticker}")
        return _dedupe_statements(records)

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

    def _get_profile(self) -> dict:
        """FMP company profile (sector, industry, ...). Current classification
        only — FMP keeps no history, so it applies to every period."""
        if self._profile is None:
            records = _get_json(f"{FMP_BASE}/profile",
                                {"symbol": self.ticker, "apikey": API_KEY})
            self._profile = dict(records[0]) if records else {}
            if self.ticker in PROFILE_OVERRIDES:
                sector, industry = PROFILE_OVERRIDES[self.ticker]
                self._profile.update(sector=sector, industry=industry)
        return self._profile

    def _get_filings(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """(period-end date, SEC filing date) pairs, one per quarterly income-statement period."""
        return sorted(
            (pd.Timestamp(item["date"]), pd.Timestamp(item["filingDate"]))
            for item in self._get_income_stmt()
        )

    def _get_daily_prices(self, symbol: str | None = None, adjusted: bool = False) -> pd.Series:
        """Daily close prices (of `symbol`, default this ticker) spanning from
        12 months before the earliest filing date through today, so forward
        (lookahead) prices are available wherever the market has caught up.
        The earliest filings' 24-month lookbacks fall outside this range and
        come back empty; the returned periods are recent enough that they are
        covered.

        adjusted=True uses dividend-adjusted closes (splits and dividends), so
        ratios of these prices are total returns rather than price returns.
        """
        symbol = symbol or self.ticker
        key = (symbol, adjusted)
        if symbol != self.ticker:
            with _BENCHMARK_LOCK:
                if key not in _BENCHMARK_PRICES:
                    _BENCHMARK_PRICES[key] = _fetch_daily_prices(symbol, _BENCHMARK_START, adjusted)
            return _BENCHMARK_PRICES[key]
        return self._get_daily_bars(adjusted)["close"]

    def _get_daily_bars(self, adjusted: bool = False) -> pd.DataFrame:
        """This ticker's daily close and volume over the _get_daily_prices range."""
        if adjusted not in self._daily_prices:
            filings = self._get_filings()
            start = (filings[0][1] - pd.DateOffset(months=12, days=10)).strftime("%Y-%m-%d")
            self._daily_prices[adjusted] = _fetch_daily_bars(self.ticker, start, adjusted)
        return self._daily_prices[adjusted]

    def _get_dollar_volume(self) -> pd.Series:
        """Daily traded value (close x volume, both split-adjusted, no
        dividend adjustment), e.g. to pick the most liquid share class."""
        bars = self._get_daily_bars(adjusted=False)
        return bars["close"] * bars["volume"]

    def _get_prices(self, symbol: str | None = None, adjusted: bool = False) -> pd.Series:
        """Average close price over the 3 trading days on/after each period's SEC
        filing date (not the fiscal period-end date), indexed by period-end date.

        The filing date is when financials actually become public, so pricing off
        it avoids look-ahead bias; averaging a few days past it smooths the
        immediate earnings-reaction jump rather than pricing off a single day.
        An index `symbol` (e.g. ^GSPC) is priced the same way off this ticker's
        filing dates.
        """
        key = (symbol, adjusted)
        if key not in self._prices:
            daily = self._get_daily_prices(symbol, adjusted)
            prices = {}
            for period_end, filing_date in self._get_filings():
                window = daily[daily.index >= filing_date].iloc[:3]
                # Same guard as the offset prices: no price if the stock wasn't trading yet.
                ok = len(window) > 0 and (window.index[0] - filing_date).days <= 7
                prices[period_end] = window.mean() if ok else None
            self._prices[key] = pd.Series(prices, dtype=float).sort_index()
        return self._prices[key]

    def _get_offset_prices(self, months: int, symbol: str | None = None,
                           adjusted: bool = False) -> pd.Series:
        """Average close over the 3 trading days on/after the date `months`
        calendar months from each period's filing date (negative = lookback,
        positive = lookahead), indexed by period-end date. This is the same
        3-day averaging (and filing-date anchor) as the Price field, so start
        and end prices of a return are measured the same way. Passing an index
        `symbol` (e.g. ^GSPC) uses this ticker's own filing dates, so the
        benchmark is priced on exactly the same dates.
        """
        key = (months, symbol, adjusted)
        if key not in self._offset_prices:
            daily = self._get_daily_prices(symbol, adjusted)
            result = {}
            for period_end, filing_date in self._get_filings():
                target = filing_date + pd.DateOffset(months=months)
                window = daily[daily.index >= target].iloc[:3]
                # Require 3 real trading days starting at the target date (a
                # week covers weekends/holidays): pre-listing lookbacks don't
                # pick up IPO-day prices, and lookaheads the market hasn't
                # reached yet (or past a delisting) stay NaN.
                ok = len(window) == 3 and (window.index[0] - target).days <= 7
                result[period_end] = window.mean() if ok else None
            self._offset_prices[key] = pd.Series(result, dtype=float).sort_index()
        return self._offset_prices[key]

    def _get_membership(self) -> pd.Series:
        """'Y'/'N' S&P 500 membership on each period's SEC filing date (when the
        report became public — the same anchor as Price), indexed by period-end date.
        """
        membership = _get_membership()
        return pd.Series({
            period_end: "Y" if membership.is_member(self.ticker, filing_date) else "N"
            for period_end, filing_date in self._get_filings()
        }).sort_index()

    def _target_date(self, period: pd.Period) -> pd.Timestamp:
        if self.freq == "Q":
            return period.end_time.normalize()
        return pd.Timestamp(year=period.end_time.year, month=self.month, day=self.day)

    def _assign_periods(self, available: pd.DatetimeIndex) -> pd.DataFrame:
        """Map each reported quarter-end to its nearest calendar target period.

        Returns one row per consecutive period from the first to the last one
        with data (index: PeriodIndex), columns `date` (the actual quarter-end,
        NaT where no quarter landed within max_offset_days) and `offset_days`.
        If two quarter-ends land on the same period, the closer one wins.
        """
        rows = []
        for d in available.sort_values():
            p0 = pd.Period(d, freq=self._period_freq)
            p = min((p0 - 1, p0, p0 + 1), key=lambda p: abs((d - self._target_date(p)).days))
            offset = (d - self._target_date(p)).days
            if abs(offset) <= self.max_offset_days:
                rows.append((p, d, offset))
        assigned = pd.DataFrame(rows, columns=["period", "date", "offset_days"])
        assigned = (assigned.loc[assigned["offset_days"].abs()
                                 .groupby(assigned["period"]).idxmin()]
                    .set_index("period"))
        full = pd.period_range(assigned.index.min(), assigned.index.max(),
                               freq=self._period_freq)
        return assigned.reindex(full)

    def _get_primitive_fields(self) -> pd.DataFrame:
        """BS/IS/CF-sourced fields only, at quarterly frequency: flow fields
        are TTM (trailing-4-quarter) aggregates, stock fields are snapshots.
        Formula/Price/PriceLookback/PriceLookahead/Membership/Profile fields are deferred to the
        calendarized frequency, where FIELD.N shifts mean "N calendar years back".
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

        # Rows are income-statement quarter-ends (what filing dates and prices
        # key on), plus any earlier balance-sheet-only quarters (the extra
        # year fetched for prior-year averages). BS/CF records are matched to
        # the nearest row within ALIGN_TOLERANCE_DAYS: FMP sometimes dates
        # the statements of one quarter a few days apart (52/53-week fiscal
        # calendars), and a record further off than that is left unmatched
        # rather than blanking the whole row.
        base = is_.index.union(bs.index[bs.index < is_.index.min()])
        tolerance = pd.Timedelta(days=ALIGN_TOLERANCE_DAYS)

        def align(col: pd.Series) -> pd.Series:
            return col.reindex(base, method="nearest", tolerance=tolerance)

        result = pd.DataFrame(index=base)
        for field_name, field_cfg in fields_cfg.items():
            source = field_cfg["Source"]
            if source == "BS":
                col = align(bs[field_cfg["API"]])
            elif source in flow_map:
                api = field_cfg["API"]
                raw_col = flow_map[source][api]
                if field_cfg.get("Aggregation") == "Quarter":
                    agg = raw_col  # single-quarter value, no TTM aggregation
                else:
                    agg = (raw_col.rolling(4, min_periods=4).mean()
                           if api in _AVERAGE_APIS else
                           raw_col.rolling(4, min_periods=4).sum())
                col = align(agg)
            else:
                continue  # Formula / Price / PriceLookback / PriceLookahead / Membership / Profile — handled in _get_fields

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

        # Every consecutive calendar period from the first to the last with
        # data; periods no quarter-end snapped to stay as all-NaN rows so
        # FIELD.N shifts stay calendar-exact, and are dropped at the end.
        assigned = self._assign_periods(primitives.index)
        periods = assigned.index
        dates = assigned["date"]

        frame = primitives.reindex(dates.values)
        frame.index = periods
        frame["_actual_period_end"] = dates
        frame["_offset_days"] = assigned["offset_days"]

        # Price / PriceLookback / PriceLookahead / Membership / Profile fields, resolved
        # at the snapped dates. API is the offset in months for the latter two.
        for field_name, field_cfg in fields_cfg.items():
            source = field_cfg["Source"]
            symbol = field_cfg.get("Symbol")
            adjusted = field_cfg.get("Adjusted") == "True"
            if source == "Price":
                s = self._get_prices(symbol, adjusted)
            elif source == "PriceLookback":
                s = self._get_offset_prices(-int(field_cfg["API"]), symbol, adjusted)
            elif source == "PriceLookahead":
                s = self._get_offset_prices(int(field_cfg["API"]), symbol, adjusted)
            elif source == "Membership":
                s = self._get_membership()
            elif source == "Profile":
                # Static company attribute (API = profile key), same in every row.
                frame[field_name] = self._get_profile().get(field_cfg["API"])
                continue
            else:
                continue
            s = s.reindex(dates.values)
            s.index = periods
            frame[field_name] = s
            rounding = field_cfg.get("Rounding")
            if rounding and rounding != "None":
                frame[field_name] = frame[field_name].round(int(rounding))

        # Formula fields, in metrics.JSON declaration order so dependencies
        # (e.g. ROE needs AVG_COMM needs COMM) resolve correctly.
        for field_name, field_cfg in fields_cfg.items():
            if field_cfg["Source"] != "Formula":
                continue
            frame[field_name] = _eval_formula(field_cfg["Formula"], frame, self._periods_per_year)
            rounding = field_cfg.get("Rounding")
            if rounding and rounding != "None":
                frame[field_name] = frame[field_name].round(int(rounding))

        return frame

    def _get_metrics(self) -> pd.DataFrame:
        fields = self._get_fields()
        result = pd.DataFrame(index=fields.index)

        for subset in METRICS_CFG["Metrics"].values():
            for metric_name, metric_cfg in subset.items():
                rounding = metric_cfg.get("Rounding")
                result[metric_name] = _eval_formula(metric_cfg["Formula"], fields, self._periods_per_year)
                if rounding and rounding != "None":
                    result[metric_name] = result[metric_name].round(int(rounding))

        # Column suffixes for backtesting: ".Y" = target (set in metrics.JSON),
        # ".D" = drop (neither predictor nor target, e.g. filters/metadata);
        # unsuffixed columns are predictors.
        result["_actual_period_end.D"] = fields["_actual_period_end"]
        result["_offset_days.D"] = fields["_offset_days"]
        # Drop calendar periods no reported quarter-end snapped to.
        result = result[result["_actual_period_end.D"].notna()]
        return result.iloc[-self.n:]


if __name__ == "__main__":
    FMP_Financial("AAPL")._get_metrics().to_csv("aapl_metrics.csv", index_label="Period")

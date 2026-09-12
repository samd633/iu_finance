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
    return eval(transformed, {"__df": df})


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
        self._prices        = None

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

    def _get_prices(self) -> pd.Series:
        if self._prices is None:
            dates = sorted(item["date"] for item in self._get_income_stmt())
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

        print("CF columns:", cf.columns.tolist())

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

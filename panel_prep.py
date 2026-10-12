"""
Data cleaning and variable manipulation on the S&P 500 panel.

Reads the panel written by index_panel.py (sp500_panel.csv) rather than
rebuilding it, and restores its (Period, Ticker) index with Period as a
quarterly pandas Period so periods sort and shift correctly.

Column suffixes follow fmp_call: unsuffixed = predictor, ".Y" = target,
".D" = drop (filters/metadata). PanelPrep adds ".R" = predictor rank bucket
(see rank()); the original predictor columns are kept alongside.

How each predictor is standardized is set per metric in metrics.JSON
("Standardize"): "Sector" = within sector each period, "Universe" = across
all in-index companies each period, "None" = left raw (required for .Y/.D).
"Universe" means the panel after the sector filters, so when the panel is
limited to one sector (sectors=["Technology"]) Sector and Universe give the
same result.

Each predictor's "Direction" in metrics.JSON says which end is better:
"Higher" or "Lower" ("None" for .Y/.D). rank() uses it so bucket 1 is always
the best.
"""
import json

import numpy as np
import pandas as pd

PANEL_FILE = "sp500_panel.csv"
METRICS_FILE = "metrics.JSON"
STANDARDIZE_OPTIONS = ("Sector", "Universe", "None")
DIRECTION_OPTIONS = ("Higher", "Lower", "None")
# Columns the panel adds in code rather than metrics.JSON (FMP_Financial._get_metrics).
META_COLUMNS = ("Period", "Ticker", "_actual_period_end.D", "_offset_days.D")

# Sectors (FMP taxonomy, Sector.D) left out of the analysis: bank, insurer and
# REIT accounts don't fit the EBITDA / cash-flow / capex / leverage metrics.
EXCLUDED_SECTORS = ("Financial Services", "Real Estate")


class PanelPrep:

    def __init__(self, tickers: list[str] | None = None, panel_file: str = PANEL_FILE,
                 exclude_sectors: tuple[str, ...] = EXCLUDED_SECTORS,
                 sectors: list[str] | None = None, n_buckets: int = 5,
                 target: str | None = "Fwd_Return_1Yr.Y", end_period: str | None = None):
        """
        tickers: restrict the panel to these tickers (None = all).
        panel_file: CSV written by IndexPanel.build().
        exclude_sectors: sectors dropped from the whole analysis (all their
            rows), before any cross-sectional statistics are computed.
        sectors: keep only these sectors (Sector.D, FMP taxonomy; None = all
            not excluded). Applied before standardization, so cross-sectional
            statistics only use these sectors.
        n_buckets: number of percentile buckets rank() assigns (5 = quintiles).
        target: prepare() drops rows where this ".Y" column is missing (None =
            keep all rows).
        end_period: last quarter kept, e.g. "2025Q2" (None = all). Ranks are
            per period, so cutting later quarters doesn't change earlier ones.
        """
        self.tickers = [t.upper() for t in tickers] if tickers else None
        self.panel_file = panel_file
        self.exclude_sectors = tuple(exclude_sectors)
        self.sectors = list(sectors) if sectors else None
        if n_buckets < 2:
            raise ValueError(f"n_buckets must be at least 2, got {n_buckets}")
        self.n_buckets = n_buckets
        if target is not None and not target.endswith(".Y"):
            raise ValueError(f"target must be a '.Y' column, got {target!r}")
        self.target = target
        self.end_period = pd.Period(end_period, freq="Q") if end_period else None
        if self.sectors is not None:
            clash = sorted(set(self.sectors) & set(self.exclude_sectors))
            if clash:
                raise ValueError(f"Sectors both selected and excluded: {clash}")

    @staticmethod
    def standardize_map(metrics_file: str = METRICS_FILE) -> dict[str, str]:
        """{metric column: "Sector" | "Universe" | "None"} from metrics.JSON."""
        return PanelPrep._metric_settings("Standardize", STANDARDIZE_OPTIONS, metrics_file)

    @staticmethod
    def direction_map(metrics_file: str = METRICS_FILE) -> dict[str, str]:
        """{metric column: "Higher" | "Lower" | "None"} from metrics.JSON.

        Also raises if a standardized/ranked metric has Direction "None".
        """
        directions = PanelPrep._metric_settings("Direction", DIRECTION_OPTIONS, metrics_file)
        standardize = PanelPrep.standardize_map(metrics_file)
        missing = [n for n, v in standardize.items() if v != "None" and directions[n] == "None"]
        if missing:
            raise ValueError(f"metrics.JSON: standardized metrics need Direction 'Higher' or 'Lower': {missing}")
        return directions

    @staticmethod
    def _metric_settings(key: str, options: tuple[str, ...], metrics_file: str) -> dict[str, str]:
        """{metric column: value of `key`} from metrics.JSON.

        Raises if a metric lacks the setting, uses a value not in `options`, or
        gives a target (.Y) / drop (.D) column anything but "None".
        """
        with open(metrics_file) as f:
            metrics = {name: cfg for group in json.load(f)["Metrics"].values()
                       for name, cfg in group.items()}
        settings, problems = {}, []
        for name, cfg in metrics.items():
            value = cfg.get(key)
            if value not in options:
                problems.append(f"{name}: {key}={value!r} (expected one of {options})")
            elif name.endswith((".Y", ".D")) and value != "None":
                problems.append(f"{name}: target/drop columns must use {key}='None'")
            settings[name] = value
        if problems:
            raise ValueError(f"metrics.JSON {key} settings:\n  " + "\n  ".join(problems))
        return settings

    def load(self) -> pd.DataFrame:
        """The panel (or the requested tickers), indexed by (Period, Ticker),
        limited to the selected sectors and without the excluded ones, up to
        end_period."""
        panel = pd.read_csv(self.panel_file)
        settings = self.standardize_map()
        unknown = [c for c in panel.columns if c not in settings and c not in META_COLUMNS]
        if unknown:
            raise ValueError(f"Panel columns not defined in {METRICS_FILE} (rebuild the panel?): {unknown}")
        excluded = panel["Sector.D"].isin(self.exclude_sectors)
        if self.tickers is not None:
            dropped = sorted(set(self.tickers) & set(panel.loc[excluded, "Ticker"]))
            if dropped:
                raise ValueError(f"Tickers in excluded sectors {self.exclude_sectors}: {dropped}")
        panel = panel[~excluded]
        if self.sectors is not None:
            unknown = sorted(set(self.sectors) - set(panel["Sector.D"]))
            if unknown:
                raise ValueError(f"Sectors not in {self.panel_file}: {unknown} "
                                 f"(available: {sorted(panel['Sector.D'].dropna().unique())})")
            panel = panel[panel["Sector.D"].isin(self.sectors)]
        if self.tickers is not None:
            missing = sorted(set(self.tickers) - set(panel["Ticker"]))
            if missing:
                raise ValueError(f"Tickers not in {self.panel_file} (or outside sectors {self.sectors}): {missing}")
            panel = panel[panel["Ticker"].isin(self.tickers)]
        panel["Period"] = pd.PeriodIndex(panel["Period"], freq="Q")
        if self.end_period is not None:
            panel = panel[panel["Period"] <= self.end_period]
        return panel.set_index(["Period", "Ticker"]).sort_index()

    def prepare(self) -> pd.DataFrame:
        """load() + rank(), then drop rows with no `target` value.

        Rows are dropped after ranking so bucket breakpoints don't depend on
        which companies later have a forward return (e.g. acquired ones).
        """
        panel = self.rank(self.load())
        if self.target is not None:
            panel = panel[panel[self.target].notna()]
        return panel

    def standardize(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Z-score each predictor per its metrics.JSON setting: (x - mean) / std
        across companies each period, within sector for "Sector".

        Mean and std come from in-index rows (In_Index.D == "Y") only and are
        applied to every row of that period/group. +/-inf (division by zero in
        a formula) is treated as missing. A group with fewer than two values
        yields NaN. "None" columns are returned unchanged.
        """
        out = panel.copy()
        in_index = out["In_Index.D"].eq("Y")
        for cols, keys in self._groupings(out):
            x = out[cols].replace([np.inf, -np.inf], np.nan)
            grouped = x.where(in_index, axis=0).groupby(keys)
            out[cols] = (x - grouped.transform("mean")) / grouped.transform("std")
        return out

    def rank(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add a "<metric>.R" column per predictor: its percentile bucket,
        1..n_buckets, across companies each period (within sector for "Sector"
        metrics). 1 = best 1/n_buckets per the metric's Direction (highest
        values for "Higher", lowest for "Lower"), n_buckets = worst.

        Bucket breakpoints come from in-index rows (In_Index.D == "Y") only and
        are applied to every row of that period/group. Ties share the bucket of
        their average rank. Missing or +/-inf values get no bucket (NaN).
        Original columns are kept unchanged.
        """
        out = panel.copy()
        in_index = out["In_Index.D"].eq("Y").to_numpy()
        directions = self.direction_map()
        ranks = {}
        for cols, keys in self._groupings(out):
            groups = list(out.groupby(keys).indices.values())
            for col in cols:
                x = out[col].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
                if directions[col] == "Lower":
                    x = -x
                buckets = np.full(len(x), np.nan)
                for idx in groups:
                    buckets[idx] = self._buckets(x[idx], x[idx][in_index[idx]], self.n_buckets)
                ranks[f"{col}.R"] = buckets
        return pd.concat([out, pd.DataFrame(ranks, index=out.index)], axis=1)

    def _groupings(self, panel: pd.DataFrame) -> list[tuple[list[str], list]]:
        """[(predictor columns, groupby keys)] per metrics.JSON "Standardize"
        setting: "Universe" groups by period, "Sector" by period and sector."""
        settings = self.standardize_map()
        period = panel.index.get_level_values("Period")
        result = []
        for setting, keys in (("Universe", [period]), ("Sector", [period, panel["Sector.D"]])):
            cols = [c for c in panel.columns if settings.get(c) == setting]
            if cols:
                result.append((cols, keys))
        return result

    @staticmethod
    def _buckets(x: np.ndarray, ref: np.ndarray, n_buckets: int) -> np.ndarray:
        """Bucket (1 = highest) of each x by its descending average rank among
        the non-missing ref values. An x not in ref ranks halfway between its
        neighbours."""
        ref = np.sort(ref[~np.isnan(ref)])
        if len(ref) == 0:
            return np.full(len(x), np.nan)
        lo = np.searchsorted(ref, x, side="left")
        hi = np.searchsorted(ref, x, side="right")
        avg_rank = (len(ref) - hi) + (hi - lo + 1) / 2
        # Small epsilon keeps exact boundaries (rank 20 of 100) in the upper bucket.
        buckets = np.clip(np.ceil(avg_rank * n_buckets / len(ref) - 1e-9), 1, n_buckets)
        return np.where(np.isnan(x), np.nan, buckets)


if __name__ == "__main__":
    prep = PanelPrep(sectors=["Technology"], end_period="2025Q2")
    df = prep.prepare()
    periods = df.index.get_level_values("Period")
    print(f"{len(df)} rows x {df.shape[1]} columns, "
          f"{df.index.get_level_values('Ticker').nunique()} tickers, sectors {prep.sectors}, "
          f"{periods.min()}-{periods.max()}, rows with {prep.target}")
    r_cols = [c for c in df.columns if c.endswith(".R")]
    counts = (df.loc[df["In_Index.D"].eq("Y"), r_cols].apply(lambda c: c.value_counts()).T
              .reindex(columns=range(1, prep.n_buckets + 1)).fillna(0).astype(int))
    counts["missing"] = df[r_cols].isna().sum()
    print(f"In-index rows per bucket, {len(r_cols)} ranked predictors:")
    print(counts.to_string())

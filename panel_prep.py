"""
Data cleaning and variable manipulation on the S&P 500 panel.

Reads the panel written by index_panel.py (sp500_panel.csv) rather than
rebuilding it, and restores its (Period, Ticker) index with Period as a
quarterly pandas Period so periods sort and shift correctly.

Column suffixes follow fmp_call: unsuffixed = predictor, ".Y" = target,
".D" = drop (filters/metadata). PanelPrep adds ".R" = predictor rank bucket
(see rank()) and ".F" = 1 if that rank was filled rather than actual (see
fill_missing()); the original predictor columns are kept alongside.

How each predictor is standardized is set per metric in metrics.JSON
("Standardize"): "Sector" = within sector each period, "Industry_Group" =
within industry group each period, "Universe" = across all in-index companies
each period, "None" = left raw (required for .Y/.D). "Universe" means the
panel after the sector filters, so when the panel is limited to one sector
(sectors=["Technology"]) Sector and Universe give the same result.

Industry groups roll FMP industries (Industry.D) up per sector, as set in
industry_groups.JSON; load() adds them as "Industry_Group.D". A sector with no
entry there is one group (its sector name); a listed sector must map every
one of its industries.

Each predictor's "Direction" in metrics.JSON says which end is better:
"Higher" or "Lower" ("None" for .Y/.D). rank() uses it so bucket 1 is always
the best.

Missing ranks are filled by fill_missing(): the bottom bucket where the
metric's optional "Bottom_If" condition (a pandas expression over panel
columns, e.g. "E_P <= 0" = currently loss-making) holds, the middle bucket
otherwise.
"""
import numpy as np
import pandas as pd

from metrics_cfg import METRICS_FILE, load_metrics_cfg

PANEL_FILE = "sp500_panel.csv"
INDUSTRY_GROUPS_FILE = "industry_groups.JSON"
TICKER_GROUPS_KEY = "Tickers"  # per-sector {ticker: group} entry in INDUSTRY_GROUPS_FILE
STANDARDIZE_OPTIONS = ("Sector", "Industry_Group", "Universe", "None")
DIRECTION_OPTIONS = ("Higher", "Lower", "None")
# Columns the panel adds in code rather than metrics.JSON (FMP_Financial._get_metrics).
META_COLUMNS = ("Period", "Ticker", "_actual_period_end.D", "_offset_days.D")

# Sectors (FMP taxonomy, Sector.D) left out of the analysis: bank, insurer and
# REIT accounts don't fit the EBITDA / cash-flow / capex / leverage metrics.
EXCLUDED_SECTORS = ("Financial Services", "Real Estate")
# Tickers left out of the analysis because FMP's sector looks wrong for them:
# TWX (Time Warner) is tagged Technology / Media & Entertainment; to be
# re-assigned to another sector later.
EXCLUDED_TICKERS = ("TWX",)
# Tickers moved to another sector (and industry) before any filtering, where
# FMP's sector doesn't match the business: ticker -> (Sector.D, Industry.D).
# Internet platforms (ad- and subscription-funded software) are tagged
# Communication Services / Internet Content & Information; GICS also had them
# in Information Technology until 2018. DASH (same FMP industry) is a delivery
# marketplace and stays where it is. CERN (Cerner, hospital health-record
# software and services) and XRX (Xerox, copiers/printers; GICS Technology
# Hardware) join the Technology groups that match their businesses, as do
# PAYX (Paychex, payroll/HR software; peer of ADP, already Technology) and TTD
# (The Trade Desk, ad-buying software platform). DASH (DoorDash, delivery
# marketplace; GICS Consumer Discretionary) joins Consumer Cyclical's services
# group. Packaging makers (FMP: Consumer Cyclical / Packaging & Containers)
# move to Basic Materials, as in GICS. Building-products makers and distributors
# (FMP: Basic Materials / Construction Materials) move to Industrials.
SECTOR_OVERRIDES = {
    **{ticker: ("Technology", "Internet Content & Information")
       for ticker in ("GOOGL", "META", "TWTR", "RDDT", "MTCH", "AABA")},
    **{ticker: ("Technology", "Software - Application") for ticker in ("PAYX", "TTD")},
    "CERN": ("Technology", "Information Technology Services"),
    "XRX": ("Technology", "Computer Hardware"),
    "DASH": ("Consumer Cyclical", "Internet Content & Information"),
    **{ticker: ("Basic Materials", "Packaging & Containers")
       for ticker in ("AMCR", "AVY", "BALL", "OI", "PKG", "SEE", "SW", "WRK")},
    **{ticker: ("Industrials", "Building Products") for ticker in ("JCI", "MAS", "FBIN", "BLDR")},
}


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
        """{metric column: "Sector" | "Industry_Group" | "Universe" | "None"} from metrics.JSON."""
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
    def bottom_if_map(metrics_file: str = METRICS_FILE) -> dict[str, str]:
        """{metric column: "Bottom_If" condition} for the metrics that set one.

        Raises if a metric that isn't ranked (Standardize "None") sets one.
        """
        metrics = {name: cfg for group in load_metrics_cfg(metrics_file)["Metrics"].values()
                   for name, cfg in group.items()}
        rules = {name: cfg["Bottom_If"] for name, cfg in metrics.items() if "Bottom_If" in cfg}
        standardize = PanelPrep.standardize_map(metrics_file)
        unranked = [n for n in rules if standardize[n] == "None"]
        if unranked:
            raise ValueError(f"metrics.JSON: Bottom_If set on metrics that aren't ranked: {unranked}")
        return rules

    @staticmethod
    def _metric_settings(key: str, options: tuple[str, ...], metrics_file: str) -> dict[str, str]:
        """{metric column: value of `key`} from metrics.JSON.

        Raises if a metric lacks the setting, uses a value not in `options`, or
        gives a target (.Y) / drop (.D) column anything but "None".
        """
        metrics = {name: cfg for group in load_metrics_cfg(metrics_file)["Metrics"].values()
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
        with SECTOR_OVERRIDES applied, limited to the selected sectors and
        without the excluded ones, up to end_period."""
        panel = pd.read_csv(self.panel_file)
        settings = self.standardize_map()
        unknown = [c for c in panel.columns if c not in settings and c not in META_COLUMNS]
        if unknown:
            raise ValueError(f"Panel columns not defined in {METRICS_FILE} (rebuild the panel?): {unknown}")
        moved = panel["Ticker"].isin(SECTOR_OVERRIDES)
        panel.loc[moved, ["Sector.D", "Industry.D"]] = [SECTOR_OVERRIDES[t] for t in panel.loc[moved, "Ticker"]]
        excluded = panel["Sector.D"].isin(self.exclude_sectors) | panel["Ticker"].isin(EXCLUDED_TICKERS)
        if self.tickers is not None:
            dropped = sorted(set(self.tickers) & set(panel.loc[excluded, "Ticker"]))
            if dropped:
                raise ValueError(f"Tickers excluded (sectors {self.exclude_sectors} "
                                 f"or EXCLUDED_TICKERS): {dropped}")
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
        panel["Industry_Group.D"] = self.industry_groups(panel)
        panel["Period"] = pd.PeriodIndex(panel["Period"], freq="Q")
        if self.end_period is not None:
            panel = panel[panel["Period"] <= self.end_period]
        return panel.set_index(["Period", "Ticker"]).sort_index()

    @staticmethod
    def industry_groups(panel: pd.DataFrame, groups_file: str = INDUSTRY_GROUPS_FILE) -> pd.Series:
        """Industry group of each row per industry_groups.JSON: the group its
        Industry.D maps to within its Sector.D, or the sector name for sectors
        not listed there. A sector's optional "Tickers" entry ({ticker: group})
        places single companies in a group regardless of their FMP industry.

        Raises if a listed sector has an industry not mapped to any group, an
        industry mapped to two groups, or a ticker placed in an unknown group.
        """
        cfg = load_metrics_cfg(groups_file)
        lookup, by_ticker, problems = {}, {}, []
        for sector, groups in cfg.items():
            groups = dict(groups)
            for ticker, group in groups.pop(TICKER_GROUPS_KEY, {}).items():
                if group not in groups:
                    problems.append(f"{sector} / {ticker}: unknown group {group!r}")
                by_ticker[sector, ticker] = group
            for group, industries in groups.items():
                for industry in industries:
                    if (sector, industry) in lookup:
                        problems.append(f"{sector} / {industry}: in both {lookup[sector, industry]} and {group}")
                    lookup[sector, industry] = group
        if problems:
            raise ValueError(f"{groups_file}:\n  " + "\n  ".join(problems))
        groups = []
        unmapped = set()
        for sector, industry, ticker in zip(panel["Sector.D"], panel["Industry.D"], panel["Ticker"]):
            if sector not in cfg:
                groups.append(sector)
            elif (sector, ticker) in by_ticker:
                groups.append(by_ticker[sector, ticker])
            elif (sector, industry) in lookup:
                groups.append(lookup[sector, industry])
            else:
                unmapped.add((sector, industry))
                groups.append(None)
        if unmapped:
            raise ValueError(f"Industries not mapped to a group in {groups_file}: "
                             + ", ".join(f"{s} / {i}" for s, i in sorted(unmapped, key=str)))
        return pd.Series(groups, index=panel.index)

    def prepare(self) -> pd.DataFrame:
        """load() + rank(), then keep only in-index rows (In_Index.D == "Y")
        with a `target` value, and fill_missing().

        Rows are dropped after ranking so bucket breakpoints don't depend on
        which companies later have a forward return (e.g. acquired ones).
        """
        panel = self.rank(self.load())
        panel = panel[panel["In_Index.D"].eq("Y")]
        if self.target is not None:
            panel = panel[panel[self.target].notna()]
        return self.fill_missing(panel)

    def fill_missing(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Fill missing predictor ranks ("<metric>.R"): bottom bucket
        (n_buckets) where the metric's "Bottom_If" condition holds on that
        row, middle bucket ((n_buckets + 1) / 2, i.e. 3 of 5) otherwise.

        Adds a "<metric>.F" flag per ranked metric: 1 where the rank was
        filled, 0 where it is the metric's actual bucket. Raw metric columns
        and the target rank are left as they are.
        """
        out = panel.copy()
        rules = self.bottom_if_map()
        middle = (self.n_buckets + 1) / 2
        flags = {}
        for col in self.standardize_map():
            name = f"{col}.R"
            if name not in out:
                continue
            missing = out[name].isna()
            bottom = out.eval(rules[col]).fillna(False).astype(bool) if col in rules else False
            out.loc[missing & bottom, name] = self.n_buckets
            out.loc[missing & ~bottom, name] = middle
            flags[f"{col}.F"] = missing.astype(int)
        return pd.concat([out, pd.DataFrame(flags, index=out.index)], axis=1)

    def standardize(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Z-score each predictor per its metrics.JSON setting: (x - mean) / std
        across companies each period, within sector for "Sector" and within
        industry group for "Industry_Group".

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
        metrics, within industry group for "Industry_Group" metrics). 1 = best 1/n_buckets per the metric's Direction (highest
        values for "Higher", lowest for "Lower"), n_buckets = worst.

        Bucket breakpoints come from in-index rows (In_Index.D == "Y") only and
        are applied to every row of that period/group. Ties share the bucket of
        their average rank. Missing or +/-inf values get no bucket (NaN).
        Original columns are kept unchanged.

        The `target` is bucketed the same way, across companies each period
        (1 = highest return), as "<target>.R.Y" (e.g. Fwd_Return_1Yr.R.Y) so
        it stays a ".Y" target column.
        """
        out = panel.copy()
        in_index = out["In_Index.D"].eq("Y").to_numpy()
        directions = self.direction_map()
        jobs = [(col, f"{col}.R", keys, directions[col])
                for cols, keys in self._groupings(out) for col in cols]
        if self.target is not None:
            jobs.append((self.target, f"{self.target.removesuffix('.Y')}.R.Y",
                         [out.index.get_level_values("Period")], "Higher"))
        ranks = {}
        for col, name, keys, direction in jobs:
            x = out[col].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            if direction == "Lower":
                x = -x
            buckets = np.full(len(x), np.nan)
            for idx in out.groupby(keys).indices.values():
                buckets[idx] = self._buckets(x[idx], x[idx][in_index[idx]], self.n_buckets)
            ranks[name] = buckets
        return pd.concat([out, pd.DataFrame(ranks, index=out.index)], axis=1)

    def _groupings(self, panel: pd.DataFrame) -> list[tuple[list[str], list]]:
        """[(predictor columns, groupby keys)] per metrics.JSON "Standardize"
        setting: "Universe" groups by period, "Sector" by period and sector,
        "Industry_Group" by period and industry group."""
        settings = self.standardize_map()
        period = panel.index.get_level_values("Period")
        result = []
        for setting, keys in (("Universe", [period]), ("Sector", [period, panel["Sector.D"]]),
                              ("Industry_Group", [period, panel["Industry_Group.D"]])):
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

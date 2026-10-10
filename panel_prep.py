"""
Data cleaning and variable manipulation on the S&P 500 panel.

Reads the panel written by index_panel.py (sp500_panel.csv) rather than
rebuilding it, and restores its (Period, Ticker) index with Period as a
quarterly pandas Period so periods sort and shift correctly.

Column suffixes follow fmp_call: unsuffixed = predictor, ".Y" = target,
".D" = drop (filters/metadata).

How each predictor is standardized is set per metric in metrics.JSON
("Standardize"): "Sector" = within sector each period, "Universe" = across
all in-index companies each period, "None" = left raw (required for .Y/.D).
"""
import json

import pandas as pd

PANEL_FILE = "sp500_panel.csv"
METRICS_FILE = "metrics.JSON"
STANDARDIZE_OPTIONS = ("Sector", "Universe", "None")
# Columns the panel adds in code rather than metrics.JSON (FMP_Financial._get_metrics).
META_COLUMNS = ("Period", "Ticker", "_actual_period_end.D", "_offset_days.D")

# Sectors (FMP taxonomy, Sector.D) left out of the analysis: bank, insurer and
# REIT accounts don't fit the EBITDA / cash-flow / capex / leverage metrics.
EXCLUDED_SECTORS = ("Financial Services", "Real Estate")


class PanelPrep:

    def __init__(self, tickers: list[str] | None = None, panel_file: str = PANEL_FILE,
                 exclude_sectors: tuple[str, ...] = EXCLUDED_SECTORS):
        """
        tickers: restrict the panel to these tickers (None = all).
        panel_file: CSV written by IndexPanel.build().
        exclude_sectors: sectors dropped from the whole analysis (all their
            rows), before any cross-sectional statistics are computed.
        """
        self.tickers = [t.upper() for t in tickers] if tickers else None
        self.panel_file = panel_file
        self.exclude_sectors = tuple(exclude_sectors)

    @staticmethod
    def standardize_map(metrics_file: str = METRICS_FILE) -> dict[str, str]:
        """{metric column: "Sector" | "Universe" | "None"} from metrics.JSON.

        Raises if a metric lacks the setting, uses an unknown value, or marks
        a target (.Y) / drop (.D) column for standardization.
        """
        with open(metrics_file) as f:
            metrics = {name: cfg for group in json.load(f)["Metrics"].values()
                       for name, cfg in group.items()}
        settings, problems = {}, []
        for name, cfg in metrics.items():
            value = cfg.get("Standardize")
            if value not in STANDARDIZE_OPTIONS:
                problems.append(f"{name}: Standardize={value!r} (expected one of {STANDARDIZE_OPTIONS})")
            elif name.endswith((".Y", ".D")) and value != "None":
                problems.append(f"{name}: target/drop columns must use Standardize='None'")
            settings[name] = value
        if problems:
            raise ValueError("metrics.JSON standardization settings:\n  " + "\n  ".join(problems))
        return settings

    def load(self) -> pd.DataFrame:
        """The panel (or the requested tickers), indexed by (Period, Ticker),
        without the excluded sectors."""
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
        if self.tickers is not None:
            missing = sorted(set(self.tickers) - set(panel["Ticker"]))
            if missing:
                raise ValueError(f"Tickers not in {self.panel_file}: {missing}")
            panel = panel[panel["Ticker"].isin(self.tickers)]
        panel["Period"] = pd.PeriodIndex(panel["Period"], freq="Q")
        return panel.set_index(["Period", "Ticker"]).sort_index()


if __name__ == "__main__":
    df = PanelPrep(["MSFT", "AAPL", "GOOGL", "NVDA", "TSLA"]).load()
    print(f"{len(df)} rows x {df.shape[1]} columns")
    print(df.groupby(level="Ticker").size().to_string())

"""
S&P 500 panel: stacks FMP_Financial's metrics table for every company in the
index over the requested window into one long DataFrame indexed by
(Period, Ticker).

The universe is survivorship-free: every company that was an S&P 500 member
at any point in the window (including since-removed and delisted ones), not
just today's constituents — see SP500Membership.universe(). Rows from before
a company joined or after it left are kept; filter on `In_Index.D` == "Y"
for the investable universe at each date.

Column suffixes follow fmp_call: unsuffixed = predictor, ".Y" = target,
".D" = drop (filters/metadata).

Every build also writes an audit CSV (AUDIT_FILE): one row per symbol that
was in the index during the window, recording how it was mapped to an FMP
ticker (renames, removals, exclusions, tail-patch approximations) and what
happened to it in the panel (rows kept, failures, delisting).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from fmp_call import FMP_Financial, _get_membership

AUDIT_FILE = "sp500_panel_audit.csv"
# Subset of the audit: every index symbol left out of the panel, and why.
EXCLUSIONS_FILE = "sp500_panel_exclusions.csv"
# A ticker whose last price is older than this is treated as no longer trading.
DELISTED_AFTER_DAYS = 10


class IndexPanel:

    def __init__(self, n: int = 40, freq: str = "Q", workers: int = 4,
                 tickers: list[str] | None = None, audit_file: str | None = AUDIT_FILE):
        """
        n, freq: passed to FMP_Financial — periods per ticker and "Q"/"A".
        workers: concurrent tickers to fetch (FMP rate limits are handled
            with retry/backoff in fmp_call).
        tickers: override the universe (e.g. for a quick test run); the
            audit then covers only those tickers' panel outcomes.
        audit_file: where build() writes the audit CSV (None to skip).
        """
        self.n = n
        self.freq = freq
        self.workers = workers
        self._tickers = tickers
        self.audit_file = audit_file
        self.failures: dict[str, str] = {}
        self.unresolved: dict[str, pd.Timestamp] = {}
        self.no_member_rows: dict[str, str] = {}  # {ticker: reason dropped}
        self.share_class_drops: dict[str, str] = {}  # {dropped class: kept class}
        self.audit: pd.DataFrame | None = None

    def _window_start(self) -> pd.Timestamp:
        """Start of the panel window: n periods back from today. Membership is
        tested on filing dates, which all fall after this date."""
        months_back = self.n * (3 if self.freq == "Q" else 12)
        return pd.Timestamp.today() - pd.DateOffset(months=months_back)

    def universe(self) -> list[str]:
        """Every ticker in the index at any point in the panel window."""
        if self._tickers is not None:
            return self._tickers
        membership = _get_membership()
        tickers = membership.universe(self._window_start())
        self.unresolved = membership.unresolved
        return tickers

    def _ticker_metrics(self, ticker: str) -> dict:
        """Metrics table plus the dates the audit needs."""
        fin = FMP_Financial(ticker, n=self.n, freq=self.freq)
        metrics = fin._get_metrics()
        daily = fin._get_daily_prices()
        dollar_volume = fin._get_dollar_volume()
        return {
            "metrics": metrics,
            "last_price": daily.index.max() if len(daily) else None,
            "first_filing": fin._get_filings()[0][1],
            "cik": fin._get_profile().get("cik") or None,
            "dollar_volume": dollar_volume[dollar_volume.index >= self._window_start()],
        }

    def _drop_extra_share_classes(self, frames: dict, results: dict) -> None:
        """Keep one share class per company (same SEC CIK, e.g. GOOG/GOOGL):
        the one with the higher average daily dollar volume over the dates
        all its classes traded (each class's own average if they never
        overlapped). The kept class is used for the company's whole history.
        Dropped classes are recorded in `self.share_class_drops`."""
        by_cik = {}
        for t in frames:
            if results[t]["cik"]:
                by_cik.setdefault(results[t]["cik"], []).append(t)
        for tickers in by_cik.values():
            if len(tickers) < 2:
                continue
            dv = pd.concat({t: results[t]["dollar_volume"] for t in tickers}, axis=1).dropna()
            means = (dv.mean() if len(dv) else
                     pd.Series({t: results[t]["dollar_volume"].mean() for t in tickers}))
            keep = means.idxmax()
            for t in tickers:
                if t != keep:
                    self.share_class_drops[t] = keep
                    del frames[t]

    def build(self) -> pd.DataFrame:
        """Fetch and stack every ticker's metrics. Per-ticker errors are
        collected in `self.failures` rather than stopping the run.

        Tickers with no In_Index "Y" row in the panel window are dropped and
        listed in `self.no_member_rows` with the reason — unless they are
        current members that simply haven't filed since joining. Companies
        with several share classes keep only the most liquid one
        (`self.share_class_drops`).
        """
        tickers = self.universe()
        _get_membership()  # build the shared membership once, before threading
        results = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._ticker_metrics, t): t for t in tickers}
            for i, fut in enumerate(as_completed(futures), 1):
                t = futures[fut]
                try:
                    results[t] = fut.result()
                except Exception as e:  # keep going; report at the end
                    self.failures[t] = f"{type(e).__name__}: {e}"
                if i % 50 == 0 or i == len(tickers):
                    print(f"  {i}/{len(tickers)} tickers ({len(self.failures)} failed)", flush=True)

        # Each ticker returns its own last n periods, so delisted companies
        # reach further back; trim everyone to the panel's common n-period window.
        last_period = max(r["metrics"].index.max() for r in results.values())
        frames = {t: r["metrics"][r["metrics"].index > last_period - self.n]
                  for t, r in results.items()}

        membership_audit = (_get_membership().audit(self._window_start())
                            if self._tickers is None else None)
        if membership_audit is not None:
            for t, f in frames.items():
                if not (f["In_Index.D"] == "Y").any():
                    reason = self._no_member_reason(t, results[t], membership_audit)
                    if reason:
                        self.no_member_rows[t] = reason
            for t in self.no_member_rows:
                del frames[t]
        frames = {t: f for t, f in frames.items() if len(f)}
        self._drop_extra_share_classes(frames, results)

        panel = pd.concat(frames, names=["Ticker", "Period"]).swaplevel().sort_index()
        self.audit = self._build_audit(panel, results, membership_audit)
        if self.audit_file:
            self.audit.to_csv(self.audit_file, index=False)
            self.audit[self.audit["panel_status"] != "included"].to_csv(
                EXCLUSIONS_FILE, index=False)
        return panel

    @staticmethod
    def _no_member_reason(ticker: str, result: dict, membership_audit: pd.DataFrame) -> str | None:
        """Why a ticker with no In_Index "Y" row in the window is dropped, or
        None to keep it (a current member that hasn't filed since joining)."""
        rows = membership_audit[membership_audit["panel_ticker"] == ticker]
        if (rows["last_in_index"] == "").any():
            return None  # current member: rows turn "Y" once it files as a member
        left = max(pd.Timestamp(d) for d in rows["last_in_index"])
        if result["first_filing"] > left:
            return (f"dropped: FMP data starts {result['first_filing'].date()}, after the "
                    f"company left the index on {left.date()} (ticker reused by a different company)")
        return (f"dropped: left the index on {left.date()}, before any filing date "
                f"in the panel window")

    def _build_audit(self, panel: pd.DataFrame, results: dict,
                     membership_audit: pd.DataFrame | None) -> pd.DataFrame:
        """Membership mapping (SP500Membership.audit) plus each panel ticker's
        outcome: CIK and other share classes, rows kept, rows in the index,
        last price date, panel status."""
        if membership_audit is not None:
            audit = membership_audit.copy()
        else:
            audit = pd.DataFrame({"symbol": self._tickers, "panel_ticker": self._tickers})

        last_price = {t: r["last_price"] for t, r in results.items()}
        cik = {t: r["cik"] for t, r in results.items() if r["cik"]}
        classes = {}
        for t, c in cik.items():
            classes.setdefault(c, []).append(t)

        tickers = panel.index.get_level_values("Ticker")
        rows = pd.Series(tickers).value_counts()
        in_index = pd.Series(tickers[panel["In_Index.D"].eq("Y").values]).value_counts()
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=DELISTED_AFTER_DAYS)

        def outcome(t):
            if not t:
                return "excluded (unresolved)"
            if t in self.failures:
                return f"failed: {self.failures[t]}"
            if t in self.no_member_rows:
                return self.no_member_rows[t]
            if t in self.share_class_drops:
                return (f"dropped: share class of {self.share_class_drops[t]} "
                        f"(lower average dollar volume)")
            if t not in in_index:
                return "included (current member; no filing since joining the index yet)"
            return "included"

        audit["cik"] = audit["panel_ticker"].map(cik).fillna("")
        audit["other_share_classes"] = audit["panel_ticker"].map(
            lambda t: ", ".join(x for x in classes.get(cik.get(t), []) if x != t))
        audit["panel_rows"] = audit["panel_ticker"].map(rows).fillna(0).astype(int)
        audit["panel_rows_in_index"] = audit["panel_ticker"].map(in_index).fillna(0).astype(int)
        audit["last_price_date"] = audit["panel_ticker"].map(
            lambda t: last_price[t].date() if last_price.get(t) is not None else "")
        audit["still_trading"] = audit["panel_ticker"].map(
            lambda t: "" if last_price.get(t) is None
            else ("Y" if last_price[t] >= cutoff else "N"))
        audit["panel_status"] = audit["panel_ticker"].map(outcome)
        return audit


if __name__ == "__main__":
    ip = IndexPanel()
    panel = ip.build()
    panel.to_csv("sp500_panel.csv")
    membership = _get_membership()
    print(f"Wrote {len(panel)} rows x {panel.shape[1]} columns "
          f"({panel.index.get_level_values('Ticker').nunique()} tickers) to sp500_panel.csv")
    print(f"Audit of {len(ip.audit)} index symbols written to {ip.audit_file}")
    print(f"Membership history file ends {membership.history_last_date.date()}")
    for note in membership.tail_notes:
        print(f"  note: {note}")
    if ip.unresolved:
        print(f"Unresolved former members (excluded): {sorted(ip.unresolved)}")
    if ip.no_member_rows:
        print(f"Dropped, no In_Index rows in window ({len(ip.no_member_rows)}):")
        for t, reason in sorted(ip.no_member_rows.items()):
            print(f"  {t}: {reason}")
    if ip.share_class_drops:
        print("Share classes dropped (kept the more liquid class): "
              + ", ".join(f"{d} -> kept {k}" for d, k in sorted(ip.share_class_drops.items())))
    if ip.failures:
        print(f"Failed ({len(ip.failures)}):")
        for t, err in sorted(ip.failures.items()):
            print(f"  {t}: {err}")

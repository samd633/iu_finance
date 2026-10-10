"""
Point-in-time S&P 500 membership.

History comes from the fja05680/sp500 GitHub dataset: one row per index
change since 1996, each listing the full constituent set *under the tickers
in use on that date* (originally from Andreas Clenow's "Trading Evolved",
maintained since 2019 from Wikipedia and press releases). That file lags
by a few weeks to months, so the tail up to today is patched from
Wikipedia's current constituents table:

  - additions are dated by the table's "Date added" column;
  - renames (old ticker dropped, new ticker added) are dated by FMP's
    symbol-change feed;
  - removals have no date on Wikipedia, so they are approximated as the
    first post-snapshot addition date (index swaps are usually same-day).
    These are reported in `tail_notes`.

Because historical tickers are as-of-date (Facebook is "FB" before
2022-06-09, "META" after), membership for a current FMP ticker is checked
against each of its former symbols over the window it held that symbol,
using FMP's symbol-change history.
"""
import bisect
import io
from pathlib import Path

import pandas as pd
import requests_cache

HISTORY_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
               "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FMP_BASE = "https://financialmodelingprep.com/stable"
HEADERS = {"User-Agent": "iu_finance-scraper/1.0 (personal research project)"}
HISTORY_LOCAL = "sp500_history_local.csv"  # fallback copy of HISTORY_URL

# Pure renames (same company, same SEC filer) missing from FMP's
# symbol-change feed, as (date, old symbol, new symbol). Add entries here
# when the panel audit shows a former member that should map to a live ticker.
MANUAL_SYMBOL_CHANGES = [
    ("2020-04-06", "ARNC", "HWM"),  # Arconic Inc. renamed Howmet Aerospace (spun off new Arconic Corp as ARNC)
]
STALE_HISTORY_DAYS = 120  # warn when the history file's last row is older than this


def _to_fmp_ticker(ticker: str) -> str:
    """Wikipedia / the history file write class shares as BRK.B; FMP expects BRK-B."""
    return ticker.strip().replace(".", "-")


class SP500Membership:

    def __init__(self, api_key: str):
        self._session = requests_cache.CachedSession(
            "fmp_cache", expire_after=86400, allowable_codes=(200,)
        )
        self._api_key = api_key
        self.tail_notes: list[str] = []
        self.unresolved: dict[str, pd.Timestamp] = {}
        self.history_last_date: pd.Timestamp | None = None
        # Tail-patch assumptions, {ticker: assumed date}, surfaced in audit().
        self.approx_removals: dict[str, pd.Timestamp] = {}
        self.assumed_adds: dict[str, pd.Timestamp] = {}
        self._symbol_changes = self._fetch_symbol_changes()
        self._dates, self._members = self._build_snapshots()
        self._alias_cache: dict[str, list[tuple]] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _get(self, url: str, **params):
        resp = self._session.get(url, params=params, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp

    def _fetch_symbol_changes(self) -> pd.DataFrame:
        records = self._get(f"{FMP_BASE}/symbol-change",
                            apikey=self._api_key, limit=100000).json()
        df = pd.DataFrame(records)[["date", "oldSymbol", "newSymbol"]]
        manual = pd.DataFrame(MANUAL_SYMBOL_CHANGES, columns=["date", "oldSymbol", "newSymbol"])
        df = pd.concat([df, manual], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df["oldSymbol"] = df["oldSymbol"].map(_to_fmp_ticker)
        df["newSymbol"] = df["newSymbol"].map(_to_fmp_ticker)
        return df.sort_values("date").reset_index(drop=True)

    def _fetch_history(self) -> list[tuple[pd.Timestamp, frozenset]]:
        """The history file from GitHub, saved locally on each successful
        download and read back from that copy if GitHub is unreachable (or
        the file has been moved/renamed upstream)."""
        try:
            text = self._get(HISTORY_URL).text
            Path(HISTORY_LOCAL).write_text(text)
        except Exception as e:
            if not Path(HISTORY_LOCAL).exists():
                raise
            text = Path(HISTORY_LOCAL).read_text()
            self.tail_notes.append(f"WARNING: history download failed ({type(e).__name__}); "
                                   f"using local copy {HISTORY_LOCAL}")
        hist = pd.read_csv(io.StringIO(text))
        return [
            (pd.Timestamp(d), frozenset(_to_fmp_ticker(t) for t in tickers.split(",")))
            for d, tickers in zip(hist["date"], hist["tickers"])
        ]

    def _fetch_current(self) -> pd.DataFrame:
        table = pd.read_html(io.StringIO(self._get(WIKI_URL).text),
                             attrs={"id": "constituents"})[0]
        return pd.DataFrame({
            "ticker": table["Symbol"].astype(str).map(_to_fmp_ticker),
            "added": pd.to_datetime(table["Date added"], errors="coerce"),
        })

    def _build_snapshots(self) -> tuple[list[pd.Timestamp], list[frozenset]]:
        """History-file snapshots, plus synthetic snapshots replaying the
        changes between its last row and today's Wikipedia list."""
        snapshots = self._fetch_history()
        last_date, last_set = snapshots[-1]
        self.history_last_date = last_date
        age = (pd.Timestamp.today() - last_date).days
        if age > STALE_HISTORY_DAYS:
            self.tail_notes.append(f"WARNING: history file ends {last_date.date()} ({age} days ago); "
                                   f"the Wikipedia-patched tail is long and removal dates are approximate")
        current = self._fetch_current()
        now_set = frozenset(current["ticker"])

        added = now_set - last_set
        removed = last_set - now_set

        events: list[tuple[pd.Timestamp, str, str]] = []  # (date, "+"/"-", ticker)

        # Renames: an old ticker that dropped out and a new one that came in,
        # linked by an FMP symbol change after the history file's last row.
        sc = self._symbol_changes
        renames = sc[(sc["date"] > last_date)
                     & sc["oldSymbol"].isin(removed) & sc["newSymbol"].isin(added)]
        for _, r in renames.iterrows():
            events += [(r["date"], "-", r["oldSymbol"]), (r["date"], "+", r["newSymbol"])]
        added -= set(renames["newSymbol"])
        removed -= set(renames["oldSymbol"])

        added_dates = current.set_index("ticker")["added"]
        for t in sorted(added):
            d = added_dates.get(t)
            if pd.isna(d) or d <= last_date:
                d = last_date + pd.Timedelta(days=1)
                self.assumed_adds[t] = d
                self.tail_notes.append(f"{t}: in current list but no post-{last_date.date()} "
                                       f"add date; assumed added {d.date()}")
            events.append((d, "+", t))

        if removed:
            add_dates = [d for d, op, _ in events if op == "+"]
            d = min(add_dates) if add_dates else pd.Timestamp.today().normalize()
            for t in sorted(removed):
                events.append((d, "-", t))
                self.approx_removals[t] = d
            self.tail_notes.append(f"removal dates unknown for {sorted(removed)}; "
                                   f"approximated as {d.date()}")

        members = set(last_set)
        for d in sorted({e[0] for e in events}):
            for _, op, t in (e for e in events if e[0] == d):
                (members.add if op == "+" else members.discard)(t)
            snapshots.append((d, frozenset(members)))

        return [d for d, _ in snapshots], [m for _, m in snapshots]

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def members_on(self, date) -> frozenset:
        """Constituent tickers (as of that date's symbols) on `date`; empty before 1996."""
        i = bisect.bisect_right(self._dates, pd.Timestamp(date)) - 1
        return self._members[i] if i >= 0 else frozenset()

    def aliases(self, ticker: str) -> list[tuple]:
        """[(symbol, valid_from, valid_to)] for a current ticker, walking FMP
        symbol changes backward. None bounds are open-ended.

        The walk stops when the symbol was last *given up* by someone else
        more recently than it was taken on (ticker recycling), so an
        unrelated earlier holder of the symbol is never attributed to us.
        """
        ticker = _to_fmp_ticker(ticker).upper()
        if ticker in self._alias_cache:
            return self._alias_cache[ticker]

        sc = self._symbol_changes
        out = []
        symbol, upper = ticker, None
        for _ in range(20):  # guard against cycles
            before = sc if upper is None else sc[sc["date"] < upper]
            taken = before[before["newSymbol"] == symbol]
            given = before[before["oldSymbol"] == symbol]
            t_date = taken["date"].max() if len(taken) else None
            g_date = given["date"].max() if len(given) else None
            if t_date is None or (g_date is not None and g_date > t_date):
                out.append((symbol, g_date, upper))
                break
            out.append((symbol, t_date, upper))
            # Several same-day changes into this symbol (e.g. MYL->VTRS and
            # the when-issued VTRSV->VTRS): prefer the one that was an index
            # member the day before.
            olds = taken.loc[taken["date"] == t_date, "oldSymbol"].tolist()
            prior = self.members_on(t_date - pd.Timedelta(days=1))
            symbol = next((s for s in olds if s in prior), olds[-1])
            upper = t_date

        self._alias_cache[ticker] = out
        return out

    def _forward_chain(self, symbol: str, as_of) -> list[tuple[str, pd.Timestamp]]:
        """FMP symbol changes after `as_of`, followed forward:
        [(new_symbol, change_date), ...] (e.g. CBS -> VIAC -> PARA)."""
        sc = self._symbol_changes
        after = pd.Timestamp(as_of)
        chain = []
        for _ in range(20):  # guard against cycles
            nxt = sc[(sc["oldSymbol"] == symbol) & (sc["date"] > after)]
            if nxt.empty:
                break
            row = nxt.iloc[0]  # earliest change after `after`
            symbol, after = row["newSymbol"], row["date"]
            chain.append((symbol, after))
        return chain

    def current_symbol(self, symbol: str, as_of) -> str:
        """Follow FMP symbol changes forward from `as_of` to the symbol's
        latest ticker (e.g. FB seen in 2020 -> META)."""
        chain = self._forward_chain(symbol, as_of)
        return chain[-1][0] if chain else symbol

    def audit(self, start, end=None) -> pd.DataFrame:
        """One row per as-of symbol in the index at any point between `start`
        and `end` (default: today), documenting how it maps into the panel
        universe and every assumption behind that:

          symbol          ticker as it appears in the membership history
          first_in_index  first date in the index under this symbol (since 1996)
          last_in_index   last date in the index under this symbol (blank if current)
          panel_ticker    FMP ticker used for its data (blank if excluded)
          status          current member / renamed / former member / unresolved
          rename_chain    FMP symbol changes after it left, e.g. "META (2022-06-09)"
          notes           tail-patch approximations and exclusion reasons

        Mapping: each symbol is followed forward from the last date it was in
        the index; the mapped ticker is used only if its own alias history
        confirms it was that member, else the original symbol is used
        (typically a removed or delisted company). Symbols that can't be
        confirmed either way — usually a delisted company whose ticker has
        since been reused, so FMP would serve the wrong company's data — are
        "unresolved" and excluded.
        """
        start = pd.Timestamp(start)
        end = pd.Timestamp(end) if end is not None else pd.Timestamp.today()
        i0 = max(bisect.bisect_right(self._dates, start) - 1, 0)
        i1 = bisect.bisect_right(self._dates, end)
        in_window = set().union(*self._members[i0:i1])

        # last = last snapshot listing the symbol; it stayed a member until
        # the day before the following snapshot (last_day).
        first, last, last_day = {}, {}, {}
        for i, (d, members) in enumerate(zip(self._dates, self._members)):
            for s in members & in_window:
                first.setdefault(s, d)
                last[s] = d
                if i + 1 < len(self._dates):
                    last_day[s] = self._dates[i + 1] - pd.Timedelta(days=1)
        current = self._members[-1]
        hist_end = self.history_last_date.date()

        rows = []
        for s in sorted(in_window):
            # Membership in the window ends at min(last, end); map from there.
            as_of = min(last[s], self._dates[i1 - 1])
            chain = self._forward_chain(s, as_of)
            t = chain[-1][0] if chain else s
            notes = []
            if s in current:
                panel, status = s, "current member"
            elif chain and self.is_member(t, as_of):
                panel, status = t, "renamed"
            elif self.is_member(s, as_of):
                panel, status = s, "former member"
                if chain:
                    notes.append(f"later symbol {t} not confirmed as the same company; "
                                 f"using {s}")
            else:
                panel, status = "", "unresolved"
                notes.append(self._unresolved_reason(s, last_day.get(s, last[s]), chain))
            if s in self.approx_removals:
                notes.append(f"removal date unknown (history file ends {hist_end}); "
                             f"approximated as {self.approx_removals[s].date()}")
            if s in self.assumed_adds:
                notes.append(f"add date unknown; assumed {self.assumed_adds[s].date()}")
            rows.append({
                "symbol": s,
                "first_in_index": first[s].date(),
                "last_in_index": "" if s in current else last_day[s].date(),
                "panel_ticker": panel,
                "status": status,
                "rename_chain": " -> ".join(f"{n} ({d.date()})" for n, d in chain),
                "notes": "; ".join(notes),
            })
        audit = pd.DataFrame(rows)
        # Other index symbols mapped to the same panel ticker (normally the
        # same company before/after a rename, e.g. FI and FISV) — listed so
        # any wrong merge of two different companies is visible.
        mapped = audit[audit["panel_ticker"] != ""]
        groups = mapped.groupby("panel_ticker")["symbol"].apply(list)
        audit["shares_panel_ticker_with"] = [
            ", ".join(x for x in groups.get(p, []) if x != s) if p else ""
            for s, p in zip(audit["symbol"], audit["panel_ticker"])
        ]
        self.unresolved = {r.symbol: last[r.symbol]
                           for r in audit.itertuples() if r.status == "unresolved"}
        return audit

    def _unresolved_reason(self, symbol: str, left: pd.Timestamp, chain: list) -> str:
        """Why an index symbol can't be matched to FMP data, naming the other
        company when the ticker was taken over by one."""
        sc = self._symbol_changes
        reused = sc[(sc["newSymbol"] == symbol) & (sc["date"] > left)]
        if len(reused):
            r = reused.iloc[0]
            return (f"excluded: ticker reused by another company ({r['oldSymbol']} renamed to "
                    f"{symbol} on {r['date'].date()}), so FMP data under {symbol} is not this company")
        if chain:
            t = chain[-1][0]
            return (f"excluded: renamed to {t}, but {t} isn't confirmed as the same company "
                    f"(ticker likely reused, or a later rename is missing from the FMP feed)")
        return "excluded: no FMP symbol history links this ticker to a current company"

    def universe(self, start, end=None) -> list[str]:
        """Panel tickers of every company in the index at any point between
        `start` and `end`, including since-removed and delisted members — the
        survivorship-free backtest universe. See audit() for the mapping;
        excluded symbols are listed in `self.unresolved`.
        """
        audit = self.audit(start, end)
        return sorted(set(audit.loc[audit["panel_ticker"] != "", "panel_ticker"]))

    def is_member(self, ticker: str, date) -> bool:
        date = pd.Timestamp(date)
        members = self.members_on(date)
        for symbol, start, end in self.aliases(ticker):
            if (start is None or date >= start) and (end is None or date < end):
                return symbol in members
        return False

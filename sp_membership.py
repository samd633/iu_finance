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

import pandas as pd
import requests_cache

HISTORY_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
               "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FMP_BASE = "https://financialmodelingprep.com/stable"
HEADERS = {"User-Agent": "iu_finance-scraper/1.0 (personal research project)"}


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
        df["date"] = pd.to_datetime(df["date"])
        df["oldSymbol"] = df["oldSymbol"].map(_to_fmp_ticker)
        df["newSymbol"] = df["newSymbol"].map(_to_fmp_ticker)
        return df.sort_values("date").reset_index(drop=True)

    def _fetch_history(self) -> list[tuple[pd.Timestamp, frozenset]]:
        hist = pd.read_csv(io.StringIO(self._get(HISTORY_URL).text))
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
                self.tail_notes.append(f"{t}: in current list but no post-{last_date.date()} "
                                       f"add date; assumed added {d.date()}")
            events.append((d, "+", t))

        if removed:
            add_dates = [d for d, op, _ in events if op == "+"]
            d = min(add_dates) if add_dates else pd.Timestamp.today().normalize()
            for t in sorted(removed):
                events.append((d, "-", t))
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
            symbol = taken.loc[taken["date"] == t_date, "oldSymbol"].iloc[-1]
            upper = t_date

        self._alias_cache[ticker] = out
        return out

    def is_member(self, ticker: str, date) -> bool:
        date = pd.Timestamp(date)
        members = self.members_on(date)
        for symbol, start, end in self.aliases(ticker):
            if (start is None or date >= start) and (end is None or date < end):
                return symbol in members
        return False

import time
import requests
import pandas as pd

class SEC_API:
    """
    Wraps SEC EDGAR API calls for a single ticker, returning clean DataFrames.

    Parameters
    ----------
    ticker   : str   — stock ticker symbol
    start    : str   — start of date range (inclusive), e.g. '2019-12-31'
    end      : str   — end of date range (inclusive)
    freq     : str   — 'Q' for quarterly (10-Q periods Q1–Q4) or
                       'K' for annual only (10-K, FY)
    cache_ttl: int|None — seconds before companyfacts cache expires;
                          None = never auto-expire; default 86400
    """

    def __init__(self, ticker: str, start: str, end: str,
                 freq: str = 'Q', cache_ttl: int | None = 86400):
        if freq not in ('Q', 'K'):
            raise ValueError("freq must be 'Q' (quarterly) or 'K' (annual)")
        self.ticker = ticker.upper()
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.freq = freq
        self.cache_ttl = cache_ttl

        self._cik: str | None = None
        self._companyfacts: dict | None = None
        self._cache_ts: float | None = None
        self.EDGAR_HEADERS = {'User-Agent': 'samd633@gmail.com'}

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_valid(self) -> bool:
        if self._companyfacts is None or self._cache_ts is None:
            return False
        if self.cache_ttl is None:
            return True
        return (time.time() - self._cache_ts) < self.cache_ttl

    def refresh(self) -> None:
        """Invalidate the companyfacts cache; next data call re-fetches."""
        self._companyfacts = None
        self._cache_ts = None

    # ------------------------------------------------------------------
    # Internal fetchers
    # ------------------------------------------------------------------

    def _fetch_cik(self) -> str:
        if self._cik is not None:
            return self._cik
        r = requests.get(
            'https://www.sec.gov/files/company_tickers.json',
            headers=self.EDGAR_HEADERS
        )
        r.raise_for_status()
        for entry in r.json().values():
            if entry['ticker'].upper() == self.ticker:
                self._cik = str(entry['cik_str']).zfill(10)
                return self._cik
        raise ValueError(f"Ticker '{self.ticker}' not found in EDGAR")

    def _fetch_companyfacts(self) -> dict:
        if self._cache_valid():
            return self._companyfacts
        cik = self._fetch_cik()
        r = requests.get(
            f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',
            headers=self.EDGAR_HEADERS
        )
        r.raise_for_status()
        self._companyfacts = r.json()
        self._cache_ts = time.time()
        return self._companyfacts

    # ------------------------------------------------------------------
    # Core helper
    # ------------------------------------------------------------------

    @property
    def _fp_filter(self) -> list[str]:
        return ['FY'] if self.freq == 'K' else ['Q1', 'Q2', 'Q3', 'Q4', 'FY']

    def get_fact_df(self, taxonomy: str, concept: str, unit: str) -> pd.DataFrame:
        """
        Extract a single XBRL concept as a date-filtered, deduplicated DataFrame.

        Deduplication keeps the most recently filed value when a period is amended.
        """
        facts = self._fetch_companyfacts()
        try:
            raw = facts['facts'][taxonomy][concept]['units'][unit]
        except KeyError:
            raise KeyError(
                f"'{taxonomy}/{concept}' with unit '{unit}' "
                f"not found in EDGAR for {self.ticker}"
            )

        df = pd.DataFrame(raw)
        df = df[df['fp'].isin(self._fp_filter)]
        if self.freq == 'K':
            df = df[df['form'].isin(['10-K', '10-K405', '10-KSB', '10-KT', '10-K/A'])]
        df['end'] = pd.to_datetime(df['end'])
        df = df[(df['end'] >= self.start) & (df['end'] <= self.end)]
        df = (
            df.sort_values('filed')
              .drop_duplicates(subset=['end', 'fp'], keep='last')
              .sort_values('end')
              .reset_index(drop=True)
        )
        return df


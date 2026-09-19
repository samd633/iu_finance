"""
Scraper for the S&P 500 constituent list, with sector classification.

Membership comes from the Wikipedia "List of S&P 500 companies" table (FMP's
constituent endpoints are not on our plan, and Vanguard/iShares block scrapers).
Each constituent is then enriched with its company name and sector via the FMP
`profile` endpoint, and the full result is written to SP500_Tick.JSON.

Tickers are normalised to FMP format (e.g. BRK.B -> BRK-B) so the list is
directly consumable by FMP downstream. The Wikipedia request is left uncached
so each scrape reflects current membership; the FMP profile lookups use the
shared `fmp_cache`, so re-scrapes only pay for newly added tickers.
"""
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import requests_cache

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FMP_BASE = "https://financialmodelingprep.com/stable"


class Scraper:

    OUTPUT_FILE = "SP500_Tick.JSON"

    # Wikipedia asks API/scraper clients to send a descriptive User-Agent.
    HEADERS = {"User-Agent": "iu_finance-scraper/1.0 (personal research project)"}

    def __init__(self):
        with open("fmp_key.txt") as f:
            self.api_key = f.read().strip()
        # Cache only the FMP profile lookups; Wikipedia membership stays live.
        self.fmp = requests_cache.CachedSession(
            "fmp_cache", expire_after=86400, allowable_codes=(200,)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_symbols(self) -> list[str]:
        """Current S&P 500 ticker symbols from the Wikipedia constituents table."""
        resp = requests.get(WIKI_URL, headers=self.HEADERS, timeout=30)
        resp.raise_for_status()
        table = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})[0]
        return table["Symbol"].dropna().astype(str).tolist()

    @staticmethod
    def _to_fmp_ticker(ticker: str) -> str:
        """Wikipedia writes class shares as BRK.B; FMP expects BRK-B."""
        return ticker.strip().replace(".", "-")

    def _fetch_profile(self, ticker: str) -> dict | None:
        """Company profile (name, sector, ...) from FMP. None if not found."""
        for attempt in range(6):
            resp = self.fmp.get(
                f"{FMP_BASE}/profile",
                params={"symbol": ticker, "apikey": self.api_key},
                timeout=30,
            )
            if resp.status_code != 429:
                break
            time.sleep(min(60, 5 * 2 ** attempt))
        resp.raise_for_status()
        data = resp.json()
        return data[0] if data else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def SP500_Scrape(self) -> list[dict]:
        """
        Fetch current S&P 500 membership, classify each company by sector via FMP,
        and overwrite SP500_Tick.JSON. Returns the list of constituent records
        (ticker, company_name, sector).
        """
        tickers = sorted({
            self._to_fmp_ticker(sym) for sym in self._fetch_symbols() if sym.strip()
        })
        if not tickers:
            raise ValueError("Wikipedia returned no constituents.")

        constituents = []
        for ticker in tickers:
            profile = self._fetch_profile(ticker)
            constituents.append({
                "ticker": ticker,
                "company_name": profile.get("companyName") if profile else None,
                "sector": profile.get("sector") if profile else None,
            })

        payload = {
            "source": "Wikipedia S&P 500 constituents; sectors via FMP",
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "count": len(constituents),
            "constituents": constituents,
        }
        Path(self.OUTPUT_FILE).write_text(json.dumps(payload, indent=2) + "\n")
        return constituents


if __name__ == "__main__":
    constituents = Scraper().SP500_Scrape()
    n_missing = sum(c["sector"] is None for c in constituents)
    print(f"Wrote {len(constituents)} constituents to {Scraper.OUTPUT_FILE}"
          + (f" ({n_missing} without a sector)" if n_missing else ""))

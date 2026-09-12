"""
Scraper for the S&P 500 constituent list, with sector classification.

Source of truth for membership is the Vanguard S&P 500 ETF (VOO). Vanguard
renders its holdings client-side, so we call the backing portfolio-holdings
JSON API directly rather than scraping the rendered page. Each constituent is
then enriched with its company name and sector via the FMP `profile` endpoint,
and the full result is written to SP500_Tick.JSON.

Tickers are normalised to FMP format (e.g. BRK.B -> BRK-B) so the list is
directly consumable by FMP downstream. The Vanguard holdings request is left
uncached so each scrape reflects current membership; the FMP profile lookups
use the shared `fmp_cache`, so re-scrapes only pay for newly added tickers.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
import requests_cache

VANGUARD_BASE = (
    "https://investor.vanguard.com/investment-products/etfs/profile"
    "/api/{product_id}/portfolio-holding/stock"
)
FMP_BASE = "https://financialmodelingprep.com/stable"


class Scraper:

    VOO_PRODUCT_ID = "0968"
    OUTPUT_FILE = "SP500_Tick.JSON"
    PAGE_SIZE = 500

    # Vanguard's API rejects requests without a browser-like User-Agent / Referer.
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
        "Referer": "https://investor.vanguard.com/investment-products/etfs/profile/voo",
    }

    def __init__(self):
        with open("fmp_key.txt") as f:
            self.api_key = f.read().strip()
        # Cache only the FMP profile lookups; Vanguard holdings stay live.
        self.fmp = requests_cache.CachedSession(
            "fmp_cache", expire_after=86400, allowable_codes=(200,)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_holdings(self) -> list[dict]:
        """Fetch all VOO holding records from Vanguard, following pagination."""
        url = VANGUARD_BASE.format(product_id=self.VOO_PRODUCT_ID)
        holdings: list[dict] = []
        start = 1
        while True:
            resp = requests.get(
                url,
                headers=self.HEADERS,
                params={"start": start, "count": self.PAGE_SIZE},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            holdings.extend(data.get("fund", {}).get("entity", []))

            total = int(data.get("size", len(holdings)))
            start += self.PAGE_SIZE
            if start > total:
                break
        return holdings

    @staticmethod
    def _to_fmp_ticker(ticker: str) -> str:
        """Vanguard writes class shares as BRK.B; FMP expects BRK-B."""
        return ticker.strip().replace(".", "-")

    def _fetch_profile(self, ticker: str) -> dict | None:
        """Company profile (name, sector, ...) from FMP. None if not found."""
        resp = self.fmp.get(
            f"{FMP_BASE}/profile",
            params={"symbol": ticker, "apikey": self.api_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data[0] if data else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def SP500_Scrape(self) -> list[dict]:
        """
        Fetch current VOO membership, classify each company by sector via FMP,
        and overwrite SP500_Tick.JSON. Returns the list of constituent records
        (ticker, company_name, sector).
        """
        holdings = self._fetch_holdings()
        if not holdings:
            raise ValueError("Vanguard API returned no holdings.")

        as_of = holdings[0].get("asOfDate", "")[:10]
        tickers = sorted({
            self._to_fmp_ticker(h["ticker"])
            for h in holdings
            if (h.get("ticker") or "").strip()
        })

        constituents = []
        for ticker in tickers:
            profile = self._fetch_profile(ticker)
            constituents.append({
                "ticker": ticker,
                "company_name": profile.get("companyName") if profile else None,
                "sector": profile.get("sector") if profile else None,
            })

        payload = {
            "source": "Vanguard VOO ETF holdings; sectors via FMP",
            "as_of_date": as_of,
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

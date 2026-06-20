import time
import requests
import requests_cache
import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import db

requests_cache.install_cache('yfinance_cache', expire_after=3600, allowable_codes=(200,))

def get_historical_splits(ticker_symbol, retries=3):
    """Fetches all historical stock splits for a given ticker symbol."""
    print(f"Fetching data for {ticker_symbol.upper()}...")

    ticker = yf.Ticker(ticker_symbol)
    time.sleep(0.5)

    df_split = None
    for attempt in range(retries):
        try:
            df_split = ticker.splits
            break
        except YFRateLimitError:
            if attempt == retries - 1:
                raise
            backoff = 2 ** (attempt + 2)  # 4s, then 8s
            print(f"Rate limited; retrying in {backoff}s...")
            time.sleep(backoff)

    if df_split is None or df_split.empty:
        print(f"No split data found for {ticker_symbol.upper()}.")
        return None

    df_split = df_split.reset_index()
    df_split.columns = ['Date', 'Split Ratio']
    df_split['Ticker'] = ticker_symbol.upper()
    df_split['Date'] = pd.to_datetime(df_split['Date']).dt.date
    df_split = df_split[df_split['Date'] >= pd.Timestamp('2005-12-31').date()]

    return df_split

def write_split(splits_data):
    rows = [
        (row['Ticker'], row['Date'], float(row['Split Ratio']))
        for _, row in splits_data.iterrows()
    ]
    sql = "INSERT INTO dbo.share_split (ticker, split_dt, split_factor) VALUES (%s, %s, %s)"
    with db.get_connection() as conn:
        cur = conn.cursor()
        cur.executemany(sql, rows)
        conn.commit()
    print(f"Wrote {len(rows)} split record(s) to dbo.share_split.")


# --- Example Usage ---
if __name__ == "__main__":
    target_ticker = "NVDA"
    splits_data = get_historical_splits(target_ticker)
    print(splits_data)
    # if splits_data is not None:
    #     write_split(splits_data)
    # else:
    #     print(f"No splits to write for {target_ticker.upper()}.")
import pandas as pd
import yfinance as yf

def get_historical_splits(ticker_symbol):
    """Fetches all historical stock splits for a given ticker symbol.

    Args:
        ticker_symbol (str): The stock ticker (e.g., 'AAPL', 'GOOGL').

    Returns:
        pd.DataFrame: A DataFrame containing the dates and split ratios.
    """
    print(f"Fetching data for {ticker_symbol.upper()}...")

    # Initialize the Ticker object
    ticker = yf.Ticker(ticker_symbol)

    # # yfinance provides a 'splits' property containing historical split data
    splits_series = ticker.history(period='1mo')

    print(splits_series)

    # if splits_series.empty:
    #     print(f"No historical stock splits found for {ticker_symbol.upper()}.")
    #     return pd.DataFrame(columns=["Date", "Split Ratio"])

    # # Convert the Series to a clean, structured DataFrame
    # splits_df = splits_series.to_frame().reset_index()

    # # Rename columns for clarity
    # splits_df.columns = ["Date", "Split Ratio"]

    # # Format the Date column to show only YYYY-MM-DD
    # splits_df["Date"] = pd.to_datetime(splits_df["Date"]).dt.date

    # # Sort with the most recent splits at the top
    # splits_df = splits_df.sort_values(by="Date", ascending=False).reset_index(
    #     drop=True
    # )

    # return splits_df


# --- Example Usage ---
if __name__ == "__main__":
    # Test with Apple (known for multiple historic splits)
    target_ticker = "AAPL"
    splits_data = get_historical_splits(target_ticker)


"""
Gross (price + dividend) total return over a given number of fiscal years.

For each fiscal year-end, return is measured from the fiscal year-end
n rows back:
    (price_end - price_start + dividends_in_period) / price_start

Dividends are summed over ex-dividend dates that fall within the window
(start_date, end_date], using adjusted dividend amounts.
"""
import pandas as pd


class TotalReturn:

    @classmethod
    def _return_for_period(
        cls,
        df: pd.DataFrame,
        prices: pd.Series,
        dividends: pd.Series,
        lookback: int,
    ) -> list[float]:
        """Compute total return for each row using `lookback` rows back as the start."""
        results = [float("nan")] * lookback

        for i in range(lookback, len(df)):
            start_date  = df["date"].iloc[i - lookback]
            end_date    = df["date"].iloc[i]
            price_start = prices.asof(start_date)
            price_end   = prices.asof(end_date)

            if pd.isna(price_start) or price_start <= 0:
                results.append(float("nan"))
                continue

            divs = dividends[
                (dividends.index > start_date) & (dividends.index <= end_date)
            ].sum()

            results.append(round((price_end - price_start + divs) / price_start, 4))

        return results

    @classmethod
    def compute(cls, df: pd.DataFrame, prices: pd.Series, dividends: pd.Series) -> pd.DataFrame:
        """
        Appends `total_return_12m` and `total_return_24m` columns to df.

        Args:
            df:        DataFrame with a 'date' column (fiscal year-end dates),
                       sorted oldest-to-newest.
            prices:    Date-indexed Series of daily closing prices.
            dividends: Date-indexed Series of adjusted dividend amounts (ex-div dates).
        """
        df = df.copy()
        df["total_return_12m"] = cls._return_for_period(df, prices, dividends, lookback=1)

        raw_24m = pd.Series(cls._return_for_period(df, prices, dividends, lookback=2))
        df["total_return_24m"] = raw_24m.apply(
            lambda r: round((1 + r) ** 0.5 - 1, 4) if pd.notna(r) else float("nan")
        ).tolist()

        return df

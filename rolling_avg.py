"""
Rolling average calculations.
"""
import pandas as pd


class RollingAvg:
    """
    Appends rolling average columns to a fundamentals DataFrame.

    Expects the DataFrame to be sorted oldest-to-newest and to already
    contain the columns being averaged.
    """

    PERIODS = [3, 5]

    @staticmethod
    def _rolling_avg(vals: pd.Series, i: int, n: int) -> float:
        """Simple average of the n values ending at position i.
        Returns NaN when fewer than n observations are available."""
        if i < n - 1:
            return float("nan")
        return round(float(vals.iloc[i - n + 1: i + 1].mean()), 4)

    @classmethod
    def compute(cls, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        For each column in `columns` and each period in PERIODS, add a
        `{col}_avg_{n}y` column to df. Returns df with the new columns appended.
        """
        df = df.copy()
        for col in columns:
            vals = df[col]
            for n in cls.PERIODS:
                df[f"{col}_avg_{n}y"] = [
                    cls._rolling_avg(vals, i, n) for i in range(len(df))
                ]
        return df

"""
CAGR (Compound Annual Growth Rate) calculations.
"""
import pandas as pd


class CAGR:
    """
    Appends rolling CAGR columns to a fundamentals DataFrame.

    Expects the DataFrame to be sorted oldest-to-newest and to already
    contain the columns being measured (e.g. eps_diluted, rev_per_share).
    """

    PERIODS = [3, 5]

    @staticmethod
    def _cagr(vals: pd.Series, i: int, n: int) -> float:
        """CAGR over n years ending at position i.
        Returns NaN for insufficient history or non-positive endpoints."""
        if i < n:
            return float("nan")
        base = vals.iloc[i - n]
        cur  = vals.iloc[i]
        if base <= 0 or cur <= 0:
            return float("nan")
        return round((cur / base) ** (1 / n) - 1, 4)

    @classmethod
    def compute(cls, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        For each column in `columns` and each period in PERIODS, add a
        `{col}_cagr_{n}y` column to df. Returns df with the new columns appended.
        """
        df = df.copy()
        for col in columns:
            vals = df[col]
            for n in cls.PERIODS:
                df[f"{col}_cagr_{n}y"] = [
                    cls._cagr(vals, i, n) for i in range(len(df))
                ]
        return df

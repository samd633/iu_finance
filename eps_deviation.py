"""
EPS standard deviation and downside deviation over a 5-year rolling window.

Both metrics operate on YoY EPS growth rates derived from 6 consecutive
EPS values (6 values → 5 growth rates). NaN is returned when there is
insufficient history or any EPS in the window is non-positive.
"""
import pandas as pd


class EPS_Deviation:

    @staticmethod
    def _yoy_growth(vals: pd.Series, i: int) -> list[float] | None:
        """5 YoY growth rates from the 6 EPS values ending at position i.
        Returns None if history is insufficient or any value is non-positive."""
        if i < 5:
            return None
        segment = vals.iloc[i - 5: i + 1]
        if any(v <= 0 for v in segment):
            return None
        return [segment.iloc[t + 1] / segment.iloc[t] - 1 for t in range(5)]

    @classmethod
    def _std_dev(cls, vals: pd.Series, i: int) -> float:
        """Population standard deviation of 5-year YoY EPS growth."""
        yoy = cls._yoy_growth(vals, i)
        if yoy is None:
            return float("nan")
        avg = sum(yoy) / 5
        sq_sum = sum((g - avg) ** 2 for g in yoy)
        return round((sq_sum / 5) ** 0.5, 4)

    @classmethod
    def _downside_dev(cls, vals: pd.Series, i: int) -> float:
        """Downside deviation of 5-year YoY EPS growth (penalises only below-average growth)."""
        yoy = cls._yoy_growth(vals, i)
        if yoy is None:
            return float("nan")
        avg = sum(yoy) / 5
        sq_sum = sum(min(g - avg, 0) ** 2 for g in yoy)
        return round((sq_sum / 5) ** 0.5, 4)

    @classmethod
    def compute(cls, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Appends `{column}_std_dev` and `{column}_downside_dev` columns to df.
        Expects df to be sorted oldest-to-newest with `column` already present.
        """
        df = df.copy()
        vals = df[column]
        df[f"{column}_std_dev"]      = [cls._std_dev(vals, i)      for i in range(len(df))]
        df[f"{column}_downside_dev"] = [cls._downside_dev(vals, i) for i in range(len(df))]
        return df

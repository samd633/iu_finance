"""
Metric calculations built on top of FMP_Financial data.
All derived metrics live here; FMP_Financial only fetches raw data.
"""
import pandas as pd
from fmp_fin import FMP_Financial
from cagr import CAGR
from rolling_avg import RollingAvg
from eps_deviation import EPS_Deviation
from total_return import TotalReturn


class Fin_Metrics:
    def __init__(self, fmp: FMP_Financial):
        self.fmp = fmp

    def get_annual_fundamentals(self) -> pd.DataFrame:
        """
        Returns one row per fiscal year with:
            fiscal_year, date,
            eps_diluted, rev_per_share,
            roce, ebitda_margin,
            price_to_cfo, price_to_earnings, price_to_book
        """
        inc = self.fmp.get_income_stmt()
        cf  = self.fmp.get_cash_flow()

        # Common equity = total stockholders' equity − preferred stock.
        # avg_common_equity uses the prior year, so we need n+1 rows then drop the oldest.
        bs = self.fmp.get_balance_sheet()
        bs["common_equity"]     = bs["totalStockholdersEquity"] - bs["preferredStock"].fillna(0)
        bs["avg_common_equity"] = (bs["common_equity"] + bs["common_equity"].shift(1)) / 2
        bs = bs.dropna(subset=["avg_common_equity"])   # drop the extra oldest row

        df = (inc.merge(bs[["date", "totalStockholdersEquity", "common_equity", "avg_common_equity"]],
                        on="date", how="left")
                 .merge(cf, on="date", how="left")
                 .sort_values("date")
                 .reset_index(drop=True))

        # Prices: asof() handles fiscal year-ends that fall on weekends / holidays.
        prices = self.fmp.get_prices()
        df["price"] = df["date"].apply(prices.asof)

        shr       = df["weightedAverageShsOutDil"]
        rev       = df["revenue"].replace(0, float("nan"))
        avg_ceq   = df["avg_common_equity"].replace(0, float("nan"))
        bvps      = (df["totalStockholdersEquity"] / shr).replace(0, float("nan"))
        cfops     = df["operatingCashFlow"] / shr
        ni_common = df["netIncome"] - df["preferredDividendsPaid"].fillna(0).abs()

        df["eps_diluted"]       = df["epsDiluted"]
        df["rev_per_share"]     = (df["revenue"] / shr).round(2)
        df["roce"]              = (ni_common / avg_ceq).round(4)
        df["ebitda_margin"]     = (df["ebitda"] / rev).round(4)
        df["price_to_cfo"]      = (df["price"] / cfops).round(2)
        df["price_to_earnings"] = (df["price"] / df["epsDiluted"].replace(0, float("nan"))).round(2)
        df["price_to_book"]     = (df["price"] / bvps).round(2)

        df = CAGR.compute(df, ["eps_diluted", "rev_per_share"])
        df = RollingAvg.compute(df, ["ebitda_margin", "roce"])
        df = EPS_Deviation.compute(df, "eps_diluted")
        df = TotalReturn.compute(df, prices, self.fmp.get_dividends())

        cols = ["fiscalYear", "date", "eps_diluted", "rev_per_share",
                "eps_diluted_cagr_3y", "eps_diluted_cagr_5y",
                "rev_per_share_cagr_3y", "rev_per_share_cagr_5y",
                "eps_diluted_std_dev", "eps_diluted_downside_dev",
                "roce", "roce_avg_3y", "roce_avg_5y",
                "ebitda_margin", "ebitda_margin_avg_3y", "ebitda_margin_avg_5y",
                "price_to_cfo", "price_to_earnings", "price_to_book",
                "total_return_12m", "total_return_24m"]
        return df[cols].rename(columns={"fiscalYear": "fiscal_year"})

    def main(self):
        print(f"{self.fmp.ticker} — last {self.fmp.n} fiscal years")
        print("=" * 80)
        print(self.get_annual_fundamentals().to_string(index=False))


if __name__ == "__main__":
    fmp = FMP_Financial("GE", n=15)
    Fin_Metrics(fmp).main()

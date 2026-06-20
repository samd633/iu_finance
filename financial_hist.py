from sec_api import SEC_API
import pandas as pd
import yfinance as yf


def _cagr(vals, i, n):
    """CAGR over n intervals ending at position i. Returns NaN for invalid inputs,
    including any non-positive endpoint (e.g. negative earnings) where a growth
    rate is meaningless."""
    if i < n:
        return float('nan')
    base = vals[i - n]
    cur = vals[i]
    if base <= 0 or cur <= 0:
        return float('nan')
    return round((cur / base) ** (1 / n), 4) - 1


def _rolling_avg(vals, i, n):
    """Simple average of n periods ending at position i."""
    if i < n - 1:
        return float('nan')
    return round(float(vals[i - n + 1:i + 1].mean()), 4)


def _eps_downside_dev(vals, i):
    """Downside deviation of YoY EPS growth over the 5 years ending at position i."""
    if i < 5:
        return float('nan')
    segment = vals[i - 5:i + 1]          # 6 EPS values → 5 YoY growth rates
    if any(v <= 0 for v in segment):     # negative/zero earnings -> growth meaningless, return NA
        return float('nan')
    yoy = [segment[t + 1] / segment[t] - 1 for t in range(5)]
    avg = sum(yoy) / 5
    devs = [min(g - avg, 0) for g in yoy] # step 4: floor at 0
    sq_sum = sum(d ** 2 for d in devs)
    return round((sq_sum / 5) ** 0.5, 4)


def _eps_std_dev(vals, i):
    """Population standard deviation of YoY EPS growth over the 5 years ending at position i."""
    if i < 5:
        return float('nan')
    segment = vals[i - 5:i + 1]          # 6 EPS values → 5 YoY growth rates
    if any(v <= 0 for v in segment):     # negative/zero earnings -> growth meaningless, return NA
        return float('nan')
    yoy = [segment[t + 1] / segment[t] - 1 for t in range(5)]
    avg = sum(yoy) / 5
    sq_sum = sum((g - avg) ** 2 for g in yoy)
    return round((sq_sum / 5) ** 0.5, 4)


class Financial_History:
    def __init__(self, ticker, start, end, freq='K'):
        if freq != 'K':
            raise NotImplementedError("Only freq='K' is currently supported")
        self.ticker = ticker
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.freq = freq
        self._splits = None
        self._yf_sess = None
        self._sec = SEC_API(
            ticker,
            self.start - pd.DateOffset(years=5),
            self.end + pd.DateOffset(years=1),
            freq=freq
        )

    def _yf_session(self):
        """Shared on-disk cached session for all yfinance calls.

        Duplicate requests within the TTL hit the local 'yfinance_cache' (SQLite)
        instead of Yahoo, which keeps repeated dev runs from breaching the rate limit.
        """
        if self._yf_sess is None:
            from requests_cache import CachedSession
            session = CachedSession('yfinance_cache', expire_after=3600)
            session.headers.update({'User-agent': 'Mozilla/5.0'})
            session.verify = False   # preserve prior curl_cffi verify=False behaviour
            self._yf_sess = session
        return self._yf_sess

    def _get_splits(self):
        """Share-split history from dbo.share_split as a tz-naive Series:
        split_date -> ratio (e.g. 4.0). Empty Series means no splits on record."""
        if self._splits is None:
            from db import get_share_splits
            self._splits = get_share_splits(self.ticker.upper())
        return self._splits

    def _split_factor(self, filed):
        """Cumulative split ratio for splits that occurred *after* each value was filed.

        A fact filed before a split is reported on the pre-split share basis; a fact
        filed after it is already split-adjusted. Keying off the filing date (not the
        period end) is what keeps already-adjusted recent years untouched while
        correcting the stale older ones. Per-share figures are divided by this factor
        and share counts are multiplied by it to land on today's basis.
        """
        splits = self._get_splits()
        filed_dt = pd.to_datetime(pd.Series(filed).reset_index(drop=True))
        if not len(splits):
            return pd.Series(1.0, index=filed_dt.index)
        factors = [splits[splits.index > f].prod() or 1.0 for f in filed_dt]
        return pd.Series(factors, index=filed_dt.index)

    def _get_eps(self):
        df = self._sec.get_fact_df('us-gaap', 'EarningsPerShareDiluted', 'USD/shares')
        factor = self._split_factor(df['filed']).values
        out = df[['end']].rename(columns={'end': 'period_end'}).copy()
        out['diluted_eps'] = (df['val'].values / factor).round(2)   # per-share: divide out post-filing splits
        return out

    def _get_revenue(self):
        # Collect all available revenue concepts; newer standards take priority on overlap.
        # Concatenation order is oldest-to-newest so drop_duplicates(keep='last') retains
        # the most current concept's value when two concepts cover the same period_end.
        frames = []
        for concept in ('SalesRevenueNet', 'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax'):
            try:
                df = self._sec.get_fact_df('us-gaap', concept, 'USD')
                frames.append(df[['end', 'val']].rename(columns={'end': 'period_end', 'val': 'revenue'}))
            except KeyError:
                continue
        if not frames:
            raise KeyError(f"No revenue concept found in EDGAR for {self.ticker}")
        return (pd.concat(frames)
                  .drop_duplicates(subset='period_end', keep='last')
                  .sort_values('period_end')
                  .reset_index(drop=True))

    def _get_shares(self):
        # Combine concepts (most-preferred last so keep='last' wins): prefer diluted then
        # basic weighted-average shares, falling back to period-end common shares only
        # where neither weighted-average exists. Combining avoids the partial-coverage
        # year gaps that first-found selection leaves when issuers switch tags.
        frames = []
        for concept in ('CommonStockSharesOutstanding',
                        'WeightedAverageNumberOfSharesOutstandingBasic',
                        'WeightedAverageNumberOfDilutedSharesOutstanding'):
            try:
                df = self._sec.get_fact_df('us-gaap', concept, 'shares')
            except KeyError:
                continue
            factor = self._split_factor(df['filed']).values
            out = df[['end']].rename(columns={'end': 'period_end'}).copy()
            out['shares'] = df['val'].values * factor   # share count: multiply by post-filing splits
            frames.append(out)
        if not frames:
            raise KeyError(f"No shares outstanding concept found in EDGAR for {self.ticker}")
        return (pd.concat(frames)
                  .drop_duplicates(subset='period_end', keep='last')
                  .sort_values('period_end')
                  .reset_index(drop=True))

    def _get_net_income(self):
        # Combine concepts rather than taking the first that exists: issuers switch tags
        # across years (e.g. LLY reports AvailableToCommonStockholdersBasic only through
        # 2012, then NetIncomeLoss). Order is least- to most-preferred so keep='last'
        # wins, preferring net income available to common (correct ROE/EPS numerator)
        # and filling gaps with total net income.
        frames = []
        for concept in ('NetIncome', 'NetIncomeLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic'):
            try:
                df = self._sec.get_fact_df('us-gaap', concept, 'USD')
                frames.append(df[['end', 'val']].rename(columns={'end': 'period_end', 'val': 'net_income'}))
            except KeyError:
                continue
        if not frames:
            raise KeyError(f"No net income concept found in EDGAR for {self.ticker}")
        return (pd.concat(frames)
                  .drop_duplicates(subset='period_end', keep='last')
                  .sort_values('period_end')
                  .reset_index(drop=True))

    def _get_gross_profit(self):
        # Prefer the directly reported GrossProfit tag.
        try:
            df = self._sec.get_fact_df('us-gaap', 'GrossProfit', 'USD')
            return df[['end', 'val']].rename(columns={'end': 'period_end', 'val': 'gross_profit'})
        except KeyError:
            pass

        # Fallback: derive gross profit = revenue - cost of goods sold. Many issuers
        # (pharma especially) never tag GrossProfit and report the two legs separately.
        # Concept order is oldest-to-newest so keep='last' prefers the current standard.
        cost_frames = []
        for concept in ('CostOfGoodsSold', 'CostOfRevenue', 'CostOfGoodsAndServicesSold'):
            try:
                c = self._sec.get_fact_df('us-gaap', concept, 'USD')
                cost_frames.append(c[['end', 'val']].rename(columns={'end': 'period_end', 'val': 'cost'}))
            except KeyError:
                continue
        if not cost_frames:
            raise KeyError(f"No gross profit (or revenue/cost) concept found in EDGAR for {self.ticker}")
        cost = (pd.concat(cost_frames)
                  .drop_duplicates(subset='period_end', keep='last')
                  .sort_values('period_end')
                  .reset_index(drop=True))

        gp = self._get_revenue().merge(cost, on='period_end', how='inner')
        gp['gross_profit'] = gp['revenue'] - gp['cost']
        return gp[['period_end', 'gross_profit']]

    def _get_book_value(self):
        # Combine concepts (most-preferred last so keep='last' wins): prefer equity
        # attributable to the parent (the right ROE denominator), filling gaps with the
        # including-noncontrolling-interest figure where the preferred tag is absent.
        frames = []
        for concept in ('StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
                        'StockholdersEquity'):
            try:
                df = self._sec.get_fact_df('us-gaap', concept, 'USD')
                frames.append(df[['end', 'val']].rename(columns={'end': 'period_end', 'val': 'book_value'}))
            except KeyError:
                continue
        if not frames:
            raise KeyError(f"No stockholders equity concept found in EDGAR for {self.ticker}")
        return (pd.concat(frames)
                  .drop_duplicates(subset='period_end', keep='last')
                  .sort_values('period_end')
                  .reset_index(drop=True))

    def _get_cfo(self):
        df = self._sec.get_fact_df('us-gaap', 'NetCashProvidedByUsedInOperatingActivities', 'USD')
        return df[['end', 'val']].rename(columns={'end': 'period_end', 'val': 'cfo'})

    def _get_prices(self):
        fetch_start = self._sec.start.strftime('%Y-%m-%d')
        fetch_end   = (self._sec.end + pd.DateOffset(days=5)).strftime('%Y-%m-%d')

        hist = yf.Ticker(self.ticker, session=self._yf_session()).history(
            start=fetch_start, end=fetch_end, auto_adjust=True
        )
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        
        return hist['Close'].sort_index()

    def growth_metrics(self):
        eps    = self._get_eps()
        rev    = self._get_revenue()
        shr    = self._get_shares()
        ni     = self._get_net_income()
        bv     = self._get_book_value()
        gp     = self._get_gross_profit()
        cfo    = self._get_cfo()
        # prices = self._get_prices()

        df = (eps.merge(rev, on='period_end', how='inner')
                 .merge(shr, on='period_end', how='inner')
                 .merge(ni,  on='period_end', how='inner')
                 .merge(bv,  on='period_end', how='inner')
                 .merge(gp,  on='period_end', how='inner')
                 .merge(cfo, on='period_end', how='inner')
                 .sort_values('period_end')
                 .reset_index(drop=True))

        df['rev_per_share'] = round(df['revenue'] / df['shares'], 4)
        df['avg_equity'] = (df['book_value'] + df['book_value'].shift(1)) / 2
        df['roe'] = df['net_income'] / df['avg_equity'].replace(0, float('nan'))
        df['gross_margin'] = df['gross_profit'] / df['revenue'].replace(0, float('nan'))

        eps_vals = df['diluted_eps'].values
        rps_vals = df['rev_per_share'].values
        roe_vals = df['roe'].values
        gm_vals  = df['gross_margin'].values

        in_window = (df['period_end'] >= self.start) & (df['period_end'] <= self.end)

        rows = []
        for i in df.index[in_window]:
            rows.append({
                'period_end':   df.at[i, 'period_end'],
                'fy':           df.at[i, 'period_end'].year,
                'diluted_eps':  df.at[i, 'diluted_eps'],
                'rev_per_share': df.at[i, 'rev_per_share'],
                'eps_cagr_5y':  _cagr(eps_vals, i, 5),
                'eps_cagr_3y':  _cagr(eps_vals, i, 3),
                'rev_cagr_5y':  _cagr(rps_vals, i, 5),
                'rev_cagr_3y':  _cagr(rps_vals, i, 3),
                'roe_3y_avg':        _rolling_avg(roe_vals, i, 3),
                'eps_downside_dev':  _eps_downside_dev(eps_vals, i),
                'eps_std_dev':       _eps_std_dev(eps_vals, i),
                # 'price_to_cfo':      prices.asof(df.at[i, 'period_end']) / (df.at[i, 'cfo'] / df.at[i, 'shares'])
                #                      if df.at[i, 'cfo'] != 0 and df.at[i, 'shares'] != 0
                #                      else float('nan'),
                # 'price_to_earnings': prices.asof(df.at[i, 'period_end']) / df.at[i, 'diluted_eps']
                #                      if df.at[i, 'diluted_eps'] != 0
                #                      else float('nan'),
                # 'price_to_book':     prices.asof(df.at[i, 'period_end']) / (df.at[i, 'book_value'] / df.at[i, 'shares'])
                #                      if df.at[i, 'book_value'] != 0 and df.at[i, 'shares'] != 0
                #                      else float('nan'),
                'gross_margin':          round(df.at[i, 'gross_margin'], 4),
                'gross_margin_3y_avg':   _rolling_avg(gm_vals, i, 3),
                'gross_margin_5y_avg':   _rolling_avg(gm_vals, i, 5),
            })

        return pd.DataFrame(rows)

    def main(self):
        print(self.ticker)
        print("=========")
        print(self.growth_metrics().to_string(index=False))


if __name__ == '__main__':
    Financial_History(
        ticker='LLY',
        start='2014-12-31',
        end='2025-12-31'
    ).main()


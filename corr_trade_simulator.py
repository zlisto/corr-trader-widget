import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Union, List, Dict, Any

warnings.filterwarnings("ignore")

class CorrTraderSimulator:
    """
    A class to simulate trading strategies based on t-statistics of asset returns.
    
    This simulator implements a correlation-based trading strategy where signals
    are generated from one asset pair and executed on another correlated pair.
    The strategy uses rolling t-statistics to identify trading opportunities and
    includes features for signal extension, trade cost modeling, and performance analysis.
    
    Attributes:
        df_open (pd.DataFrame): Hourly open prices for all trading pairs
        df_volume_usd (pd.DataFrame): Daily USD volume for all trading pairs  
        df_returns (pd.DataFrame): Hourly returns for all trading pairs
        df_cum (pd.DataFrame): Cumulative returns for all trading pairs
        pairs (pd.Index): List of available trading pairs
        rbtc (pd.Series): Bitcoin returns series
        cum_btc (pd.Series): Cumulative Bitcoin returns
        btc_return (float): Total Bitcoin return over the period
    """

    def __init__(self, start_date: str = '2025-04-10', stop_date: str = '2025-06-18'):
        """
        Initialize the CorrTraderSimulator with market data and trading parameters.
        
        Loads hourly price and volume data from CSV files, filters to the specified
        trading period, and calculates returns and cumulative returns for analysis.
        
        Parameters:
            start_date (str): Start date for trading simulation (YYYY-MM-DD format)
            stop_date (str): End date for trading simulation (YYYY-MM-DD format)
            
        Raises:
            AssertionError: If date ranges are invalid or data files are missing
            
        Note:
            Data files are expected to be in the 'data/' directory with naming
            convention: kraken_hourly_[type]_[initial_date]_to_[final_date].csv
        """
        
        initial_date = '2025-04-10' #filename initial date
        final_date = '2025-07-20'  #filename final date
        assert pd.to_datetime(start_date) < pd.to_datetime(stop_date), "Initial date must be before final date"
        assert pd.to_datetime(initial_date) <= pd.to_datetime(start_date), "Initial date must be before start date"
        assert pd.to_datetime(stop_date) <= pd.to_datetime(final_date), "Stop date must be before final date"

        
        df_open = pd.read_csv(f'data/kraken_hourly_open_{initial_date}_to_{final_date}.csv', index_col=0, parse_dates=True)
        df_volume = pd.read_csv(f'data/kraken_hourly_volume_{initial_date}_to_{final_date}.csv', index_col=0, parse_dates=True)

        print(f"Data loaded from {initial_date} to {final_date}")
        print(f"Data contains {len(df_open)} hours and {len(df_open.columns)} pairs")
        print(f"Trading from {start_date} to {stop_date}")
        ind = (df_open.index >= start_date) & (df_open.index <= stop_date)
        df_open = df_open[ind]
        df_volume = df_volume[ind]
        print(f"Trading from {df_open.index[0]} to {df_open.index[-1]}")

         # Calculate returns and cumulative returns

        df_returns = df_open.pct_change().fillna(0)
        df_cum = np.cumprod(1+df_returns)

        df_volume_usd_hour = df_volume*df_open  #compute hourly volume in USD
        df_volume_usd_day = df_volume_usd_hour.rolling(24).sum().fillna(0)  #compute daily volume in USD

        self.df_open = df_open
        self.df_volume_usd = df_volume_usd_day
        self.df_returns = df_returns
        self.df_cum  = df_cum
        self.pairs = self.df_open.columns
        # Ensure we're working with pandas Series for BTC data
        btc_series = self.df_open['XXBTZUSD']
        self.rbtc = btc_series.pct_change().fillna(0)
        self.cum_btc = (1 + self.rbtc).cumprod() - 1        
        self.btc_return = self.cum_btc.iloc[-1]
        self.pairs = self.df_returns.columns
    
    def compute_tstat(self, r: pd.Series, window: int) -> pd.Series:
        """
        Compute rolling t-statistics for a single asset based on returns.
        
        Uses a robust volatility estimator based on quantiles to compute t-statistics
        that are less sensitive to outliers than traditional standard deviation.
        
        Parameters:
            r (pd.Series): Return series for a single asset
            window (int): Rolling window size in hours for t-stat calculation
            
        Returns:
            pd.Series: Rolling t-statistics for the asset
            
        Note:
            The t-statistic is computed as: t = (mean / sigma) * sqrt(window)
            where sigma is estimated using the interquartile range method
        """
        p= 0.25
        z = 1.349
        mu = r.rolling(window).mean()
        qlow = r.rolling(window).quantile(p)
        qhigh = r.rolling(window).quantile(1-p)
        sigma = (qhigh - qlow) / z
        epsilon=1e-6
        tstat = mu / (sigma + epsilon) * np.sqrt(window)
        return tstat.fillna(0)

    def compute_df_tstat(self, window: int) -> pd.DataFrame:
        """
        Compute rolling t-statistics for all assets in the dataset.
        
        Applies the t-statistic calculation to all trading pairs simultaneously,
        using robust volatility estimation based on quantiles.
        
        Parameters:
            window (int): Rolling window size in hours for t-stat calculation
            
        Returns:
            pd.DataFrame: T-statistics for all assets with same index and columns as df_returns
            
        Note:
            This method is more efficient than calling compute_tstat() for each asset
            individually when working with the entire dataset.
        """
        qlow = 0.25
        qhigh = 1-qlow
        z = 1.349
        mu = self.df_returns.rolling(window).mean()
        qlow = self.df_returns.rolling(window).quantile(qlow)
        qhigh = self.df_returns.rolling(window).quantile(qhigh)
        sigma1 = (qhigh - qlow) / z
        sigma2 =self.df_returns.rolling(window).std()

        #median = self.df_returns.rolling(window).median()
        #mad = self.df_returns.rolling(window).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        #sigma_robust = mad / 0.6745  # Normal approximation

        
        # Use sigma2 where sigma1 is zero or very small
        epsilon=1e-6
        sigma = sigma1.mask(sigma1 < epsilon, sigma2)

        # Still guard against any remaining near-zero values
        #sigma = sigma.replace(0, epsilon).fillna(epsilon)
        sigma = sigma1
        df_tstat = mu / (sigma1 + epsilon) * np.sqrt(window)
        return df_tstat.fillna(0)

    @staticmethod
    def max_drawdown(series: pd.Series) -> float:
        """
        Calculate the maximum drawdown of a cumulative return series.
        
        Maximum drawdown is the largest peak-to-trough decline in the series,
        expressed as a percentage of the peak value.
        
        Parameters:
            series (pd.Series): Cumulative return series (should be monotonically increasing)
            
        Returns:
            float: Maximum drawdown value (negative number, e.g., -0.25 for 25% drawdown)
            
        Example:
            If a strategy goes from 100% to 75% and back to 90%, the max drawdown is -25%
        """
        peak = series.cummax()
        drawdown = (series - peak) / peak
        return drawdown.min()

    def extend_signal(self, signal_series: pd.Series, extension_hours: int = 12) -> pd.Series:
        """
        Extend binary trading signals to maintain positions for a specified duration.
        
        When a buy signal (1) occurs, this method extends the signal for the next
        N hours to simulate holding the position, preventing rapid in-and-out trading.
        
        Parameters:
            signal_series (pd.Series): Binary signal series with 0/1 values and datetime index
            extension_hours (int): Number of hours to extend each signal (default 12)
            
        Returns:
            pd.Series: Extended signal series with same index as input
            
        Note:
            This helps reduce trading frequency and associated costs by maintaining
            positions for a minimum duration after entry signals.
        """
        # Create a copy to avoid modifying the original
        signal = signal_series.copy()
        
        # Find where signal changes from 0 to 1 (entry points)
        entry_points = (signal == 1) & (signal.shift(1) == 0)
        
        # For each entry point, extend the signal by extension_hours
        for entry_idx in signal[entry_points].index:
            # Find the position of this entry point
            entry_pos = signal.index.get_loc(entry_idx)
            
            # Calculate the end position (entry + extension_hours, but don't exceed series length)
            end_pos = min(entry_pos + extension_hours + 1, len(signal))
            
            # Set the next extension_hours to 1
            signal.iloc[entry_pos:end_pos] = 1
        
        return signal
        
    def simulate_trading(self, pair1: str, pair2: str, window_tstat: int, tstat_enter: float, trade_fee: float) -> None:
        """
        Simulate correlation-based trading between two asset pairs.
        
        Generates trading signals based on t-statistics of pair1 and executes trades
        on pair2, accounting for correlation between the assets and trading costs.
        
        Parameters:
            pair1 (str): Asset pair used to generate trading signals (signal generator)
            pair2 (str): Asset pair to trade based on pair1 signals (trade target)
            window_tstat (int): Rolling window size for t-statistic calculation
            tstat_enter (float): T-statistic threshold for entering trades
            trade_fee (float): Trading fee as a percentage (e.g., 0.001 for 0.1%)
            
        Returns:
            None: Results are stored in instance attributes:
                - self.df: DataFrame with trading results and performance metrics
                - self.df_trades: DataFrame with individual trade details
                
        Note:
            The strategy buys pair2 when pair1's t-statistic exceeds the threshold,
            leveraging the correlation between the assets for predictive trading.
        """
        self.pair1 = pair1
        self.pair2 = pair2
        r1: pd.Series = self.df_returns[pair1]
        r2: pd.Series = self.df_returns[pair2]
        rho = np.corrcoef(r1,r2)[0,1]

        r1future: pd.Series = r1.shift(-1).fillna(0)
        r2future: pd.Series = r2.shift(-1).fillna(0)
        t1 = self.compute_tstat(r1, window_tstat)
        #t2 = self.compute_tstat(r2, window_tstat)
        self.tstat_pair1 = t1

        signal = (t1>tstat_enter).astype(int)
        extension_hours = 1
        signal = self.extend_signal(signal, extension_hours)
        trades = signal.diff().fillna(0)
        rsignal = r2future*signal - trade_fee*trades.abs()

        cum1 = np.cumprod(1+r1future)-1
        cum2 = np.cumprod(1+r2future)-1
        cum_signal = np.cumprod(1+rsignal)-1

        df = pd.DataFrame({'returns_pair1':r1future,
                           'returns_pair2':r2future,
                           'returns_pair2_signal': rsignal,
                           'cum_pair1':cum1,
                           'cum_pair2':cum2,
                           'cum_pair2_signal':cum_signal,                           
                           'trades':trades,
                           'signal':signal,
                           'tstat_pair1':t1,
                           'price_pair1':self.df_open[pair1],
                           'price_pair2':self.df_open[pair2],})
        self.df = df
        s = self.df.signal
        r2 = self.df.returns_pair2
        index = s.index
        s_shifted = s.shift(1, fill_value=0)
        start_times = index[(s == 1) & (s_shifted != 1)]
        end_times = index[(s == 1) & (s.shift(-1, fill_value=0) != 1)]

        trades = []
        for start, end in zip(start_times, end_times):
            r_period = r2.loc[start:end]
            returns = (1 + r_period).prod() - 1
            idx_after_end = index.get_indexer([end], method="pad")[0] + 1
            sell_time = index[idx_after_end] if idx_after_end < len(index) else end
            age_hours = (sell_time - start).total_seconds() / 3600
            trades.append({
                "buy_datetime": start,
                "sell_datetime": sell_time,
                "returns": returns,
                "age_hours": age_hours
            })

        self.df_trades = pd.DataFrame(trades)
        #self.trading_summary()
        
    def sharpe_ratio(self, r: pd.Series) -> float:
        """
        Calculate the annualized Sharpe ratio for a return series.
        
        The Sharpe ratio measures risk-adjusted returns by dividing excess returns
        by the standard deviation of returns, then annualizing the result.
        
        Parameters:
            r (pd.Series): Return series (assumed to be hourly returns)
            
        Returns:
            float: Annualized Sharpe ratio
            
        Note:
            Assumes hourly data and annualizes by multiplying by sqrt(24*365)
        """
        return r.mean()/r.std()*np.sqrt(24*365)
    
    def trading_summary(self) -> None:
        """
        Generate and print a comprehensive summary of trading simulation results.
        
        Displays key performance metrics including cumulative returns, Sharpe ratios,
        correlation statistics, and conditional expected returns for both the signal
        strategy and buy-and-hold benchmarks.
        
        Returns:
            None: Results are printed to console
            
        Note:
            This method should be called after simulate_trading() to analyze results.
            Provides comparison between signal-based trading and simple buy-and-hold strategies.
        """
        r1 = self.df.returns_pair1
        r2 = self.df.returns_pair2
        rsignal = self.df.returns_pair2_signal
        cum1 = self.df.cum_pair1.iloc[-1]
        cum2 = self.df.cum_pair2.iloc[-1]
        cum_signal = self.df.cum_pair2_signal.iloc[-1]
        rho = np.corrcoef(r1,r2)[0,1]
        rho12_lag = np.corrcoef(r1, r2.shift(-1).fillna(0))[0,1]
        rho21_lag = np.corrcoef(r2, r1.shift(-1).fillna(0))[0,1]
        mu2 = r2.mean()
        mu2_0 = r2[self.df.signal==0].mean()
        mu2_1 = r2[self.df.signal==1].mean()
        ntrades = self.df.trades.abs().sum()


        print(f"Cum Signal {self.pair2} = {cum_signal*100:.2f}%")
        print(f"Cum HODL {self.pair2} = {cum2*100:.2f}%")
        print(f"Cum HODL {self.pair1} = {cum1*100:.2f}%")

        print(f"Sharpe ratio Signal {self.pair2} = {self.sharpe_ratio(rsignal):.2f}")
        print(f"Sharpe ratio HODL {self.pair2} = {self.sharpe_ratio(r2):.2f}")        
        print(f"Sharpe ratio HODL {self.pair1} = {self.sharpe_ratio(r1):.2f}")

        print(f"{ntrades} trades")
        print(f"Corr({self.pair1},{self.pair2}) = {rho:.4f}")
        print(f"Corr({self.pair1},{self.pair2} future) = {rho12_lag:.4f}")
        print(f"Corr({self.pair2},{self.pair1} future) = {rho21_lag:.4f}")
        print(f"{self.pair2}: E[r] = {mu2:.3e}, E[r|0]={mu2_0:.3e}, E[r|1]= {mu2_1:.3e}")
    
    def plot_trades(self) -> None:
        """
        Create a visualization of trading performance and signal points.
        
        Plots cumulative returns for both the signal strategy and buy-and-hold
        benchmarks, along with markers indicating when trades were executed.
        
        Returns:
            None: Displays matplotlib plot
            
        Note:
            This method should be called after simulate_trading() to visualize results.
            Red dots indicate periods when the trading signal was active (position held).
        """
        c = self.df.cum_pair2
        cs = self.df.cum_pair2_signal
        s = self.df.signal
        plt.figure(figsize = (10,4))
        plt.plot(self.df.cum_pair2, label = f"HODL {self.pair2}", color = 'blue')
        plt.plot(self.df.cum_pair2_signal, label = f"Signal {self.pair2}", color = 'orange')
        plt.plot(self.df.cum_pair1, label = f"HODL {self.pair1}", color = 'black')
        plt.scatter(c[s==1].index, c[s==1], color = 'red', s = 2)
        plt.scatter(cs[s==1].index, cs[s==1], color = 'red', s = 2)

        #plt.scatter(c[s==0].index, c[s==0], color = 'blue', s = 4)
        plt.grid()
        plt.legend()
        plt.show()

    def optimize_pairs(self, window_tstat: int, tstat_enter: float, trade_fee: float, corr_threshold: float = 0.25) -> None:
        """
        Find optimal trading pairs based on correlation and performance metrics.
        
        Tests all possible pair combinations and identifies the best signal generator
        (pair1) for each trade target (pair2) based on cumulative returns and
        correlation thresholds.
        
        Parameters:
            window_tstat (int): Rolling window size for t-statistic calculation
            tstat_enter (float): T-statistic threshold for entering trades
            trade_fee (float): Trading fee as a percentage
            corr_threshold (float): Minimum correlation threshold for pair consideration (default 0.25)
            
        Returns:
            None: Results are stored in instance attributes:
                - self.df_pairs: DataFrame with all pair combinations and their metrics
                - self.df_pairs_opt: DataFrame with optimal pair combinations
                
        Note:
            For each pair2, finds the pair1 that generates the highest cumulative returns
            while meeting the correlation threshold. Results are sorted by performance.
        """
        results = []
        for pair2 in self.pairs:
            r2: pd.Series = self.df_returns[pair2]
            cum_max = -np.inf
            pair1_best = None
            
            for pair1 in self.pairs:
                r1: pd.Series = self.df_returns[pair1]
                rho = np.corrcoef(r1, r2.shift(-1).fillna(0))[0, 1]
                if rho >= corr_threshold:
                    self.simulate_trading(pair1, pair2, 
                                          window_tstat, 
                                          tstat_enter, 
                                          trade_fee)
                    cum_signal = self.df.cum_pair2_signal.iloc[-1]
                    cum_hodl = self.df.cum_pair2.iloc[-1]
                    
                    sharpe_signal = self.sharpe_ratio(self.df.returns_pair2_signal)
                    sharpe_hodl = self.sharpe_ratio(self.df.returns_pair2)

                    #volume_pair1 = self.df_volume_usd[pair1].mean()*24
                    volume_pair2 = self.df_volume_usd[pair2].mean()*24
                    results.append({'pair1': pair1, 
                                    'pair2': pair2, 
                                    'cum_signal': cum_signal,
                                    'cum_hodl': cum_hodl,
                                    'volume_pair2': volume_pair2,
                                    'sharpe_signal': sharpe_signal,
                                    'sharpe_hodl': sharpe_hodl,
                                    'correlation': rho,  
                                    })
                    if (cum_signal > cum_max):
                        cum_max = cum_signal
                        pair1_best = pair1
            if pair1_best is not None:
                print(f"Best pair for {pair2} (volume {volume_pair2:,.0f} USD/24hr) is {pair1_best} with cum_signal = {cum_max*100:.2f}% vs cum_hodl = {cum_hodl*100:.2f}%")
                        

        df_pairs = pd.DataFrame(results)
        df_pairs.sort_values(by='cum_signal', ascending=False, inplace=True)
        # Step 1: Keep rows where cum_signal is max for each pair2
        df_max_signal = df_pairs.loc[df_pairs.groupby('pair2')['cum_signal'].idxmax()]

        # Step 2: Among those, keep the row(s) with the best pair1 (assuming 'best' means max)
        self.df_pairs_opt = df_max_signal.loc[df_max_signal.groupby('pair2')['pair1'].idxmax()]
        self.df_pairs = df_pairs.reset_index(drop=True)




 
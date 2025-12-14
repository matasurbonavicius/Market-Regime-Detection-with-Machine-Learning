# Regional imports
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Set, Union
import matplotlib.pyplot as plt
from datetime import datetime
from fredapi import Fred
import yfinance as yf
import pandas as pd
import numpy as np
import logging


class MarketDataModule:
    
    """
    A class to fetch market data and calculate various indicators, 
    manipulate data.
    """

    def __init__(self, symbols_yfinance: List[str] = None, symbols_fred: List[str] = None, timeframe: str = 'Weekly') -> None:
        
        
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info("Initializing MarketDataModule.")
        
        if not any([symbols_yfinance, symbols_fred]):
            raise ValueError("Please pass either symbols_yfinance or symbols_fred")

        self.current_indicators_names: Set[str] = set()
        self.dataframes: Optional[pd.DataFrame] = None
        self.symbols_yfinance: List[str]        = symbols_yfinance
        self.symbols_fred: List[str]            = symbols_fred
        
        self.indicators: dict = {
            'RSI': self._compute_rsi,
            'CCI': self._compute_cci,
            'ATR': self._compute_atr,
            'MA': self._compute_ma,
            'EMA': self._compute_ema,
            'Stoch': self._compute_stochastic,
            'BB_Width': self._compute_bollinger_bands,
            'MACD': self._compute_macd,
            'Liquidity': self._compute_liquidity_indicator,
            'Spread': self._compute_spread,
            'ADX': self._compute_directional_indicator,
            'Change': self._compute_change,
            'MaxDrawdown': self._compute_max_drawdown,
            'Momentum': self._compute_momentum,
            'StocksAboveAverage': self._compute_stocks_over_average
        }

        self.timeframe = timeframe

        self.timeframe_map = {
            "Daily"  : {"Yfinance": "1d" , "FRED": "D"},
            "Weekly" : {"Yfinance": "1wk", "FRED": "W"},
            "Monthly": {"Yfinance": "1m" , "FRED": "M"}
        }

    def fetch_data_yfinance(self, start_date: str, end_date: str = None, only_close = False) -> None:

        """
        Fetch data for the given symbols.
        """
        
        tf = self.timeframe_map[self.timeframe]['Yfinance']

        self._logger.info(
            f"Fetching {tf} data from {start_date} to {end_date} for symbols: {self.symbols_yfinance}"
            )

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        for idx, symbol in enumerate(self.symbols_yfinance):
            data = yf.download(
                symbol, 
                start=start_date, 
                end=end_date, 
                interval=tf, 
                progress=False
            )

            if only_close:
                data = data['Close']
                data.name = f"{data.name}_{symbol}"
            else:
                data.rename(columns=lambda x: f"{x}_{symbol}", inplace=True)

            if self.dataframes is None:
                self.dataframes = data
            else:
                self.dataframes = pd.merge(self.dataframes, 
                                           data, 
                                           left_index=True, 
                                           right_index=True, 
                                           how='outer'
                                           )
    
    def fetch_data_fred(self, api_key: str, start_date: str, end_date: str = None) -> None:
        
        """
        Fetch data from FRED.

        Parameters:
        - api_key: Your FRED API key
        - start_date: Start date for the data
        - end_date: End date for the data (defaults to today)
        - frequency: Data frequency ("D" for daily, "W" for weekly, "M" for monthly)
        """

        tf = self.timeframe_map[self.timeframe]['FRED']
        
        self._logger.info(
            f"Fetching {tf} data from {start_date} to {end_date} for symbols: {self.symbols_fred}"
        )

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        for idx, symbol in enumerate(self.symbols_fred):
            fed = Fred(api_key)
            data = fed.get_series(symbol, start=start_date, end=end_date)

            if tf == "W":
                # Weekly data resampled to end on Saturdays
                data = data.resample('W-SAT').last()
            elif tf == "M":
                # Monthly data with end of period aggregation
                data = data.resample('M').last()

            data.name = f"{symbol}"

            data = data[start_date:]

            if self.dataframes is None:
                self.dataframes = data
            else:
                self.dataframes = pd.merge(self.dataframes, 
                                           data, 
                                           left_index=True, 
                                           right_index=True, 
                                           how='outer'
                                           )

    def get_data(self) -> pd.DataFrame:
        
        """
        Return the self.dataframes dataframe.
        """

        return self.dataframes

    def save_data(self, dataframe = None, path="market_data_module_saved.csv") -> None:

        """
        Saves the dataframe in the current folder
        """
    
        if not dataframe:
            self.dataframes.to_csv(path)
        else:
            self.dataframe.to_csv(path)
    
    def get_indicators(self) -> List[str]:
        self._logger.info(f"[get_indicators] Indicators: {list(self.current_indicators_names)}")

        """
        Return the names of the currently computed indicators.
        """

        return list(self.current_indicators_names)
    
    def scale_data(self, data_input: Union[np.ndarray, pd.DataFrame, pd.Series], only_dropna: bool = False) -> np.ndarray:
        """
        Drop NA values (if pandas object) and scale the data to have mean=0 and st.dev=1.

        Parameters:
        - data_input (pd.DataFrame | pd.Series | np.ndarray): The data to be scaled.
        - only_dropna (bool): if true, the don't scale data

        Returns:
        - np.ndarray: Scaled data with mean=0 and standard deviation=1.
        """
        index_values = None

        # Check if input is a pandas DataFrame or Series
        if isinstance(data_input, (pd.DataFrame, pd.Series)):
            data_input = data_input.dropna()
            index_values = data_input.index

            if isinstance(data_input, pd.Series):  # If Series, make it 2D
                data_input = data_input.dropna()
                data_input = data_input.values.reshape(-1, 1)
            else:
                data_input = data_input.values
                data_input = data_input[~np.isnan(data_input).any(axis=1)]
        
        elif isinstance(data_input, np.ndarray):
            if len(data_input.shape) == 1:  # If 1D numpy array
                data_input = data_input.reshape(-1, 1)
        else:
            raise ValueError("Input type not recognized. Expected pandas DataFrame/Series or numpy array.")

        # Scale the data
        scaler = StandardScaler()

        if not only_dropna:
            return scaler.fit_transform(data_input), index_values
        else:
            return data_input, index_values


    def calculate_indicators(self, indicator_list: List[str], symbol: Optional[str] = None, *args, **kwargs) -> None:
        
        """
        Calculate the requested indicators 
        and add them to the main dataframe.
        """
        
        # If symbol is None, infer it from the columns in the dataframe
        if symbol is None:
            available_columns = self.dataframes.columns
            possible_columns = [col for col in available_columns if 'Close' in col or 'None' in col]
            
            if not possible_columns:
                raise ValueError("Unable to determine the symbol column from the dataframe.")
            
            symbol = possible_columns[0]

        for indicator in indicator_list:
            if indicator in self.indicators:

                print(f'indicator: {indicator} - symbol: {symbol} - args: {args} | {kwargs}')
                self.indicators[indicator](symbol, *args, **kwargs)

                if indicator not in self.current_indicators_names:
                    self.current_indicators_names.add(f"{indicator}_{symbol.split('_')[-1]}_{str(kwargs.get('window'))}")

    def _compute_rsi(self, symbol: str, *args, **kwargs) -> None:
        
        """
        Calculates RSI indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe
        """

        window = kwargs.get('window', 14)
        symbol_suffix = symbol.split('_')[-1]
        delta = self.dataframes[symbol].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        self.dataframes['RSI_'+symbol_suffix+'_'+str(window)] = rsi
    
    def _compute_cci(self, symbol: str, *args, **kwargs) -> None:
        
        """
        Calculates RSI indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe
        """

        window = kwargs.get('window', 20)
        symbol_suffix = symbol.split('_')[-1]
        TP = (self.dataframes['High_'+symbol_suffix] + 
              self.dataframes['Low_'+symbol_suffix] + 
              self.dataframes[symbol]) / 3
        rolling_mean = TP.rolling(window=window).mean()
        rolling_std = TP.rolling(window=window).std()
        self.dataframes['CCI_'+symbol_suffix+'_'+str(window)] = (
                TP - rolling_mean) / (0.015 * rolling_std
                )
        
    def _compute_atr(self, symbol: str, return_values: bool = False, *args, **kwargs) -> None:
        """
        Calculates ATR indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe.
        If return_values is True, it returns the ATR values without adding them to the dataframe.
        """
        window = kwargs.get('window', 14)
        ticker = symbol.split('_')[-1]
        high = self.dataframes[f"High_{ticker}"].ffill()
        low = self.dataframes[f"Low_{ticker}"].ffill()
        close = self.dataframes[symbol]
        atr = high - low
        atr = atr.where(atr > (high - close).abs(), high - close)
        atr = atr.where(atr > (low - close).abs(), low - close)
        atr = atr.ffill()
        atr_values = atr.rolling(window=window).mean()

        if return_values:
            return atr_values
        else:
            self.dataframes[f"ATR_{ticker}"+'_'+str(window)] = atr_values


    def _compute_ma(self, symbol: str, *args, **kwargs) -> None:
        
        """
        Calculates SMA indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe
        """

        window = kwargs.get('window', 14)
        calculate_distance = kwargs.get('calculate_distance', True)

        ticker = symbol.split('_')[-1]
        close = self.dataframes[symbol]

        if calculate_distance:
            self.dataframes[f"MA_{ticker}"+'_'+str(window)] = close - close.rolling(window=window).mean()
        else:
            self.dataframes[f"MA_{ticker}"+'_'+str(window)] = close.rolling(window=window).mean()

    def _compute_ema(self, symbol: str, *args, **kwargs) -> None:
        
        """
        Calculates EMA indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe
        """

        window = kwargs.get('window', 14)
        calculate_distance = kwargs.get('calculate_distance', True)

        ticker = symbol.split('_')[-1]
        close = self.dataframes[symbol]

        if calculate_distance:
            self.dataframes[f"EMA_{ticker}"+'_'+str(window)] = close - close.ewm(span=window, adjust=False).mean()
        else:
            self.dataframes[f"EMA_{ticker}"+'_'+str(window)] = close.ewm(span=window, adjust=False).mean()

    def _compute_stochastic(self, symbol: str, *args, **kwargs) -> None:
        
        """
        Calculates Stochastic indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe
        """

        window = kwargs.get('window', 14)
        ticker = symbol.split('_')[-1]
        high = self.dataframes[f"High_{ticker}"].ffill()
        low = self.dataframes[f"Low_{ticker}"].ffill()
        close = self.dataframes[symbol]
        low_min = low.rolling(window=window).min()
        high_max = high.rolling(window=window).max()
        self.dataframes[f"Stoch_{ticker}"+'_'+str(window)] = 100 * (close - low_min) / (high_max - low_min)

    def _compute_bollinger_bands(self, symbol: str, *args, **kwargs) -> None:
        
        """
        Calculates BB indicator for the given symbol and adds it as a
        column to the self.dataframes dataframe
        """

        window = kwargs.get('window', 14)
        close = self.dataframes[symbol]
        ticker = symbol.split('_')[-1]
        ma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        self.dataframes[f"BBUpper_{ticker}"+'_'+str(window)] = ma + (std * 2)
        self.dataframes[f"BBLower_{ticker}"+'_'+str(window)] = ma - (std * 2)
        self.dataframes[f"BB_Width_{ticker}"+'_'+str(window)] = (ma + (std * 2)) - (ma - (std * 2))

    def _compute_macd(self, symbol: str, *args, **kwargs) -> None:
        
        """
        Calculates MACD indicator for the given symbol and adds it as columns
        to the self.dataframes dataframe. The columns added are: MACD_{symbol}, MACDSignal_{symbol},
        and MACDWidth_{symbol}.
        """

        short_window = kwargs.get('short_window', 12) 
        long_window = kwargs.get('window', 26)
        signal_window = kwargs.get('signal_window', 9)
        close = self.dataframes[symbol]
        short_ema = close.ewm(span=short_window, adjust=False).mean()
        long_ema = close.ewm(span=long_window, adjust=False).mean()
        
        macd = short_ema - long_ema
        macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
        
        macd_width = macd - macd_signal
        
        ticker = symbol.split('_')[-1]
        self.dataframes[f"MACD_{ticker}"+'_'+str(long_window)] = macd
        self.dataframes[f"MACDSignal_{ticker}"+'_'+str(long_window)] = macd_signal
        self.dataframes[f"MACDWidth_{ticker}"+'_'+str(long_window)] = macd_width

    def _compute_liquidity_indicator(self, symbol: str, *args, **kwargs):

        """
        This is a simple liquidity indicator based on the cfa institute article:
        
        https://rpc.cfainstitute.org/en/research/cfa-digest/2015/02/
            a-practical-approach-to-liquidity-calculation-digest-summary
        
        To define liquidity, the authors answer the question, 
        What amount of money is needed to create a daily single unit price
        fluctuation of the stock? They estimate the liquidity measure as the 
        ratio of volume traded multiplied by the closing price divided 
        by the price range from high to low, for the whole trading day, 
        on a logarithmic scale.
        """

        ema_period = kwargs.get('ema_period', 20)
        ticker = symbol.split('_')[-1]
        high = self.dataframes[f"High_{ticker}"]
        low = self.dataframes[f"Low_{ticker}"]
        volume = self.dataframes[f"Volume_{ticker}"]
        close = self.dataframes[symbol]
        price_range = high - low
        
        self.dataframes[f"Liquidity_Indicator_Raw_{ticker}"]     \
            = (volume * close) / price_range
        self.dataframes[f"Liquidity_Indicator_Log_{ticker}"]     \
            = np.log(self.dataframes[f"Liquidity_Indicator_Raw_{ticker}"])
        self.dataframes[f"Liquidity_{ticker}"] \
            = self.dataframes[f"Liquidity_Indicator_Raw_{ticker}"].ewm(span=ema_period, adjust=False).mean() # Exp moving average

    def _compute_spread(self, *args, **kwargs):

        """
        Takes two symbols and creates a spread from them
        """
    
        symbol1 = kwargs.get('symbol1')
        symbol2 = kwargs.get('symbol2')
        
        if not symbol1 or not symbol2:
            raise ValueError("Both symbol1 and symbol2 must be provided.")
    
        symbol1_df = self.dataframes[symbol1].ffill()
        symbol2_df = self.dataframes[symbol2].ffill()  
        self.dataframes[f"Spread {symbol1} - {symbol2}"] = symbol1_df - symbol2_df
    
    def _compute_directional_indicator(self, symbol: str, *args, **kwargs) -> None:
        
        """
        Calculates ADX, +DI and -DI for the given symbol and adds them as
        columns to the self.dataframes dataframe
        """
        
        window = kwargs.get('window', 14)
        ticker = symbol.split('_')[-1]

        # Compute True Range
        high = self.dataframes[f'High_{ticker}']
        low = self.dataframes[f'Low_{ticker}']
        close = self.dataframes[symbol]
        
        TR = high.combine(close.shift(), max) - low.combine(close.shift(), min)
        
        # Compute +DM and -DM
        move_up = high - high.shift()
        move_down = low.shift() - low
        pos_DM = np.where(((move_up > move_down) & (move_up > 0)), move_up, 0)
        neg_DM = np.where(((move_down > move_up) & (move_down > 0)), move_down, 0)
        
        # Smooth the true range and DMs using the Wilder's method (Wilder's moving average)
        TR_smoothed = TR.rolling(window=window).mean()
        pos_DM_smoothed = pd.Series(pos_DM, index=TR.index).rolling(window=window).mean()
        neg_DM_smoothed = pd.Series(neg_DM, index=TR.index).rolling(window=window).mean()
        
        # Compute +DI and -DI
        pos_DI = 100 * (pos_DM_smoothed / TR_smoothed)
        neg_DI = 100 * (neg_DM_smoothed / TR_smoothed)

        # Compute ADX
        dx = 100 * abs(pos_DI - neg_DI) / (pos_DI + neg_DI)
        adx = dx.rolling(window=window).mean()

        # Save to dataframe
        # self.dataframes['+DI_' + symbol_suffix] = pos_DI
        # self.dataframes['-DI_' + symbol_suffix] = neg_DI
        self.dataframes[f'ADX_{ticker}'+'_'+str(window)] = adx
    
    def _compute_change(self, symbol: str, *args, **kwargs) -> None:

        """
        Simply calculates rolling change from the last two periods
        """

        window = kwargs.get('window', 1)
        ticker = symbol.split('_')[-1]
        df = self.dataframes[symbol]
        pct_changes = df.pct_change()
        rolling_pct_changes = pct_changes.rolling(window=window).mean()


        self.dataframes[f'Change_{ticker}_{str(window)}'] = rolling_pct_changes

    def _compute_max_drawdown(self, symbol: str, *args, **kwargs) -> None:
        
        """
        Calculates trailing or total max drawdown for a given symbol.
        If `max` is True, it calculates the total max drawdown for all data.
        If `max` is False, it calculates the max drawdown over a specified trailing window.
        Adds the result as a column to the self.dataframes dataframe.
        """
        
        window = kwargs.get('window', 14)
        calculate_total = kwargs.get('max', False)
        ticker = symbol.split('_')[-1]
        close = self.dataframes[symbol]
        
        if calculate_total:
            # Calculate total max drawdown for all data
            rolling_max = close.expanding().max()
            daily_drawdown = close / rolling_max - 1.0
            max_drawdown = daily_drawdown.expanding().min()
            self.dataframes[f"MaxDrawdown_{ticker}"] = max_drawdown
        else:
            # Calculate trailing max drawdown for specified window
            rolling_max = close.rolling(window=window, min_periods=1).max()
            daily_drawdown = close / rolling_max - 1.0
            max_drawdown = daily_drawdown.rolling(window=window, min_periods=1).min()
            self.dataframes[f"MaxDrawdown_{ticker}_{str(window)}"] = max_drawdown

    def _compute_momentum(self, symbol: str, *args, **kwargs) -> None:
        """
        Calculates the Momentum indicator for the given symbol and adds it as
        a column to the self.dataframes dataframe.
        """
        
        window = kwargs.get('window', 14)  # Default window period is 14
        ticker = symbol.split('_')[-1]
        close = self.dataframes[symbol]
        
        momentum = close - close.shift(window)
        self.dataframes[f"Momentum_{ticker}"+'_'+str(window)] = momentum
    
    def _compute_stocks_over_average(self, symbol: str, *args, **kwargs) -> None:
        
        """
        Reads a csv file from 'Files/stocks_over_average.csv' and adds its data
        to the self.dataframes dictionary.
        """
        
        window = kwargs.get('window', 4)  # Default window period is 4 - a month if weekly data
        ticker = symbol.split('_')[-1]
        filepath = 'Files/stocks_over_average.csv'
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        self.dataframes[f"StocksAboveAverage_{ticker}_{str(window)}"] = None
        df_good = pd.concat([df[str(window)], self.dataframes[f"StocksAboveAverage_{ticker}_{str(window)}"]]).sort_index().ffill().dropna()
        self.dataframes[f"StocksAboveAverage_{ticker}_{str(window)}"] = df_good










# # Usage Example:
# # symbols = ["BAMLC0A4CBBB"]
# # api_key = "b48eef00ef6d0de74f782e6165ac7454"
# symbols_fred = ["BAMLH0A0HYM2", "BAMLC0A4CBBB", "BAMLC0A2CAA", "T10Y2Y", "AMERIBOR", "DTB3"]
# #symbols_yfinance=["^VIX", "^MOVE", "EURUSD=X", "SPY"]
# symbols_yfinance = ["SPY"]
# analyzer = MarketDataModule(symbols_fred=symbols_fred, symbols_yfinance=symbols_yfinance)
# # analyzer.fetch_data_fred(api_key, "2000-01-01")
# analyzer.fetch_data_yfinance("2000-01-01")
# analyzer.calculate_indicators(['RSI'], "Close_SPY")
# # analyzer.calculate_indicators(['Liquidity'], "Close_SPY")
# # analyzer.calculate_indicators(['Spread'], symbol1="AMERIBOR", symbol2="DTB3")
# data = analyzer.get_data()
# print(analyzer.get_indicators())
# analyzer.save_data()

# data = data.ffill()
# print(data.tail(20))

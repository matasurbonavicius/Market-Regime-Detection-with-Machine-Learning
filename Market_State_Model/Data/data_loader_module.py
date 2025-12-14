from sklearn.preprocessing import StandardScaler
from collections import namedtuple
import pandas as pd
from typing import List
import logging
import numpy as np

from Data.market_data_module import MarketDataModule
from Data.triple_barrier_labeling import Triple_Barrier_Labeling

logger = logging.getLogger(__name__)

class DataLoaderModule:

    """
    Module to handle data loading, combination, and preparation for model training and prediction.

    Attributes
    ----------
    main_df : pd.DataFrame
        The main DataFrame that combines data from different sources.
    loaded_dfs : List[pd.DataFrame]
        A list of DataFrames that have been loaded.
    indicators : List[str]
        List of indicator names.
    DataSplit : namedtuple
        A named tuple for organizing training, validation, and testing datasets.

    Methods
    -------
    load_data(path: str) -> None:
        Loads data from the given path.
    load_indicators(symbols_yfinance: list, symbols_fred: list, timeframe: str, indicators: list, underlying: str, start_date: str, end_date: str) -> None:
        Loads market indicators from given sources.
    combine_data() -> None:
        Combines data from loaded DataFrames into main_df.
    load_targets(triple_label: bool, change: bool, HMM_States: bool, on_column: str, shift_target: bool) -> None:
        Loads target values for the model into the main_df.
    prepare_data(training_period: slice, validation_period: slice, testing_period: slice, underlying: str) -> namedtuple:
        Prepares and organizes datasets for training, validation, and testing.
    """

    def __init__(self) -> None:
        
        """
        Initializes the class
        """

        self.main_df     = pd.DataFrame()
        self.loaded_dfs  = []
        self.indicators  = []

        self.labels_used = False

        self.DataSplitWithLabels = namedtuple('DataSplitWithLabels', ['training_labels'    , 'validation_labels'    , 'testing_labels'    ,
                                                                      'training_underlying', 'validation_underlying', 'testing_underlying',
                                                                      'training_features'  , 'validation_features'  , 'testing_features',
                                                                      'training_dates'     , 'validation_dates'     , 'testing_dates'])
        
        self.DataSplitNoLabels = namedtuple('DataSplitNoLabels', ['training_underlying', 'validation_underlying', 'testing_underlying',
                                                                  'training_features'  , 'validation_features'  , 'testing_features',
                                                                  'training_dates'     , 'validation_dates'     , 'testing_dates'])

    def load_data(self, path: str = "Data/HMM_data.csv") -> None:
        
        """
        Loads data from a CSV file located at the specified path.

        Parameters
        ----------
        path : str, optional
            Path to the CSV file to load. Default is "Data/HMM_data.csv".
        """
        
        logging.info(f"[load data] Loading data from {path}")
        
        data = pd.read_csv(path)

        # Data needs to have a DataTime index
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = data['Dates']
                data.drop(columns=['Dates'], inplace=True)
            except Exception as e:
                logging.error(f"[load data] Local Data DataFrame index is not DateTime & there is no column 'Dates'")
                return

        if not data.empty:
            self.loaded_dfs.append(data)

        logging.info("[load data] Local Data DataFrame Loaded. Used: ffill to forward Fill for NaN values ; dropna() to drop the first rows with leftover NaNs")

    def load_indicators(
        self, 
        symbols_yfinance: list = None, 
        symbols_fred:     list = None, 
        timeframe:        str  = "Weekly", 
        indicators:       dict = None, 
        underlying:       str  = None, 
        start_date:       str  = "2000-01-01", 
        end_date:         str  = "2023-09-01"
        ) -> None:

        """
        Loads market indicators based on provided lists of symbols and other specifications.

        Parameters
        ----------
        symbols_yfinance : list, optional
            List of symbols to fetch data from Yahoo Finance.
        symbols_fred : list, optional
            List of symbols to fetch data from Federal Reserve Economic Data (FRED).
        timeframe : str, optional
            Timeframe of data, default is "Weekly".
        indicators : dict, optional
            List of indicators to calculate.
        underlying : str, optional
            Underlying column name to calculate indicators.
        start_date : str, optional
            Start date for fetching data, default is "2000-01-01".
        end_date : str, optional
            End date for fetching data, default is "2023-01-01".
        """
        
        # Initialize MarketDataModule given by what symbols we want
        self.market_data = None
        if symbols_yfinance and symbols_fred:
            self.market_data = MarketDataModule(symbols_yfinance=symbols_yfinance, timeframe = timeframe)
            self.market_data.fetch_data_yfinance(start_date = start_date, end_date = end_date)
            self.market_data.fetch_data_fred(start_date = start_date, end_date = end_date)
            logging.info(f"[load indicators] Loaded Market Data w/ These Columns: {list(self.market_data.get_data().columns)}")
        
        elif symbols_yfinance:
            self.market_data = MarketDataModule(symbols_yfinance=symbols_yfinance, timeframe = timeframe)
            self.market_data.fetch_data_yfinance(start_date = start_date, end_date = end_date)
            logging.info(f"[load indicators] Loaded Market Data w/ These Columns: {list(self.market_data.get_data().columns)}")
        
        elif symbols_fred:
            self.market_data = MarketDataModule(symbols_yfinance=symbols_yfinance, timeframe = timeframe)
            self.market_data.fetch_data_fred(start_date = start_date, end_date = end_date)
            logging.info(f"[load indicators] Loaded Market Data w/ These Columns: {list(self.market_data.get_data().columns)}")
        
        try:
            for key, item in  indicators.items():
                for x in indicators[key]:
                    self.market_data.calculate_indicators([key], underlying, window = x)

            self.indicators = self.market_data.get_indicators()
            logging.info(f"[load indicators] Indicators Loaded Successfully.")
        except Exception:
            raise KeyError("[load indicators] Indicators were not loaded successfully. Check if underlying column is in loaded data")
            
        data = self.market_data.get_data().ffill().dropna()
        logging.info("[load indicators] Market Data DataFrame Loaded. Used:  ffill to forward Fill for NaN values ; dropna() to drop the first rows with leftover NaNs")

        if not data.empty:
            self.loaded_dfs.append(data)
        
    def combine_data(self) -> None:

        """
        Combines the loaded DataFrames into the main DataFrame, main_df.
        """

        logging.info("[combine data] Combining the DataFrames")
        date_intervals = []
        for df in self.loaded_dfs:

            print(df)
            
            df.index = pd.to_datetime(df.index)
            date_interval = df.index.to_series().dropna()
            date_interval = date_interval.diff().min()
            
            if date_interval not in date_intervals:
                date_intervals.append(date_interval)
            
        if len(date_intervals) == 1:
            logging.info("[combine data] Timeframe of all the dataframes is the same")
            for df in self.loaded_dfs:
                if not self.main_df.empty:
                    self.main_df = pd.merge(self.main_df.copy(), df, how='left', right_index=True, left_index=True)
                else:
                    self.main_df = df
        else:
            logging.info("[combine data] Timeframe of all the dataframes is not the same")
            idx = date_intervals.index(min(date_intervals)) # Index of dataframe with smallest timeframe
            main_df = self.loaded_dfs[idx]

            for df in self.loaded_dfs:
                self.main_df = pd.concat([self.main_df, df]).sort_index()

    def load_labels(
        self, 
        triple_label: bool = False, 
        change:       bool = False, 
        HMM_States:   bool = False, 
        on_column:    str  = 'Close_SPY', 
        shift_target: bool = True
        ) -> None:

        """
        Loads target values based on specified methods into the main DataFrame.

        Parameters
        ----------
        triple_label : bool, optional
            Whether to use the Triple Barrier Labelling method, default is False.
            * if market rose or fell +-2%, then label is +1/-1 accordingly. If neither, then label is 0
        change : bool, optional
            Whether to use the Change (previous-current) Labelling method, default is False.
            * If market went up, label=1 ; if market went down, label=0
        HMM_States : bool, optional
            Whether to use HMM States as labeling, default is False.
        on_column : str, optional
            The column name to base the targets on, default is 'Close_SPY'.
        shift_target : bool, optional
            Whether to shift target values, default is True.
        """

        logging.info("[load Labels] Loading the target into the main df")
        self.labels_used = True

        if not triple_label and not change and not HMM_States:
            raise KeyError("[load Labels] Must specify a method with which to load the targets")

        if triple_label:
            logging.info("[load Labels] Using the Triple Barrier Labelling")
            try:
                self.main_df = Triple_Barrier_Labeling(self.main_df, on_column).label_data()
            except Exception:
                raise KeyError("[load targets] Triple Barrier Method was not successful")
        
        elif change:
            logging.info("[load Labels] Using the Change (previous-current) Labelling")
            try:
                self.main_df['Labels'] = np.where(self.main_df[on_column].pct_change() > 0, 1, 0)
            except Exception:
                raise KeyError("[load Labels] Change was not successful")
            
        elif HMM_States:

            """
            Logic behind:

            We can assume that HMMM (or other classification algorithm) determined state at the end of the period
            is the correct state of the market.

            However we only have features for that period. To get an estimate for whats going to happen the next week,
            we should use displace the features by one period forward. 

            This way we will look into features that were 1 period old and compare them to the current state of the market.

            Goal is to train the algorithm to PREDICT, not to classify.
            
            """

            logging.info('HMM States labelling is currently not supported')

            logging.info("[load Labels] Using the HMM States as Labelling")
            
            try:
                targets_df = pd.concat([self.main_df['trained_values'], self.main_df['Predicted_values']]).sort_index().dropna()
                self.main_df['Labels'] = targets_df
            except Exception:
                raise KeyError("[load targets] HMM States were not successful]")
        
        if shift_target:
            self.main_df['Labels'] = self.main_df['Labels'].shift(-1)

    def prepare_data(
        self, 
        training_period: slice   = slice('2000-01-01', '2018-01-01'), 
        validation_period: slice = slice('2018-01-01', '2020-01-01'), 
        testing_period: slice    = slice('2020-01-01', '2023-09-01'), 
        underlying: str          = 'Close_SPY', 
        add_features: List[str]  = None,
        scale_features: bool     = False
        ) -> namedtuple:
        
        """
        Prepares and organizes the data into training, validation, and testing datasets.

        Parameters
        ----------
        training_period : slice, optional
            Period slice for training data, default is slice('2005-01-01', '2015-01-01').
        validation_period : slice, optional
            Period slice for validation data, default is slice('2015-01-01', '2020-01-01').
        testing_period : slice, optional
            Period slice for testing data, default is slice('2020-01-01', '2022-01-01').
        underlying : str, optional
            Underlying column name to be included in the datasets, default is 'Close_SPY'.

        Returns
        -------
        DataSplit : namedtuple
            A named tuple containing split datasets for training, validation, and testing.
            * types- np arrays *
        """

        logging.info("[prepare data] Preparing the datasets for training and predicting.")
        
        try:
            features = self.indicators.copy()
            
            try:
                if add_features:
                    features.extend(add_features)
                    self.main_df[add_features]
            except Exception:
                print('Printing main dataframe column names below:')
                print(list(self.main_df.columns))
                raise KeyError("[prepare data] add_features are not in main dataframe. Check syntax")

            features_extended = features.copy()

            if self.labels_used:
                features_extended.extend([underlying, 'Labels'])
            else:
                features_extended.extend([underlying])

            model_data            = features_extended
            logging.debug('[prepare data] Step 1 Completed')

            # Split data
            training_data         = self.main_df.loc[training_period]
            validation_data       = self.main_df.loc[validation_period]
            testing_data          = self.main_df.loc[testing_period]
            logging.debug('[prepare data] Step 2 Completed')

            print("------**-")
            print(training_data)
            print(model_data)

            # Deal w/ NaNs
            training_dataset      = training_data[model_data].ffill().dropna()
            validation_dataset    = validation_data[model_data].ffill().dropna()
            testing_dataset       = testing_data[model_data].ffill().dropna()
            logging.debug('[prepare data] Step 3 Completed')

            # Prepare labels
            if 'Labels' in training_dataset:
                training_labels       = np.array(training_dataset['Labels'])
                validation_labels     = np.array(validation_dataset['Labels'])
                testing_labels        = np.array(testing_dataset['Labels'])
            logging.debug('[prepare data] Step 4 Completed')

            # Prepare underlying
            training_underlying   = np.array(training_dataset[underlying])
            validation_underlying = np.array(validation_dataset[underlying])
            testing_underlying    = np.array(testing_dataset[underlying])
            logging.debug('[prepare data] Step 5 Completed')

            # Prepare features
            training_features     = np.array(training_dataset[features])
            validation_features   = np.array(validation_dataset[features])
            testing_features      = np.array(testing_dataset[features])
            logging.debug('[prepare data] Step 6 Completed')

            # Prepare dates
            training_dates     = np.array(training_dataset.index)
            validation_dates   = np.array(validation_dataset.index)
            testing_dates      = np.array(testing_dataset.index)
            logging.debug('[prepare data] Step 6 Completed')

            if scale_features:
                scaler = StandardScaler()
                training_features     = scaler.fit_transform(training_features)
                validation_features   = scaler.fit_transform(validation_features)
                testing_features      = scaler.fit_transform(testing_features)
        
        except Exception:
            raise KeyError('[prepare data] Data preparation failed')

        if self.labels_used:
            return self.DataSplitWithLabels(training_labels    , validation_labels    , testing_labels    ,
                                            training_underlying, validation_underlying, testing_underlying,
                                            training_features  , validation_features  , testing_features  ,
                                            training_dates     , validation_dates     , testing_dates)
        else:
            return self.DataSplitNoLabels(  training_underlying, validation_underlying, testing_underlying,
                                            training_features  , validation_features  , testing_features  ,
                                            training_dates     , validation_dates     , testing_dates)

    def get_indicators(self) -> None:
        self.market_data.get_indicators()

# # Usage Example
# loader = DataLoaderModule()
# symbols_yfinance = ['SPY']

# #                  |       Reversal     |  |   Volatility  |   |   Trend   |   | Liquidity |
# #indicators       = ['RSI', 'CCI', 'Stoch', 'ATR', 'BB_Width',  'MACD', 'ADX',  'Liquidity']

# indicators       = ['Liquidity']
# timeframe        = 'Weekly'
# underlying       = "Close_SPY"

# loader.load_indicators(symbols_yfinance=symbols_yfinance, indicators=indicators, timeframe=timeframe, underlying = underlying)
# loader.combine_data()
# loader.load_labels(change=True, shift_target=True)
# data = loader.prepare_data()

# print(data.training_features)

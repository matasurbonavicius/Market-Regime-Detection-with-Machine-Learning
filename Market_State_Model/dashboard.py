from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Data.data_loader_module import DataLoaderModule
from Helpers.plotting_module import Plotting_Module
from Helpers.backtesting import Backtest
from Helpers.oracle_labeling import Oracle_Labeler

from Unsupervised_Learning.unsupervised_manager import Unsupervised_Learning_Manager
from Unsupervised_Learning.DTW_classificator import classify_states

from Supervised_Learning.supervised_manager import Supervised_Training_Manager
from Supervised_Learning.XGBoost_model import XGBoost_Model


DataMapped = namedtuple('DataMapped', ['ensembled', 'colors_mapped_HMM', 'colors_mapped_KMeans', 'colors_mapped_ensembled'])


# ---- >>>> Classification <<<< ----

def get_data() -> namedtuple:
    
    """
    Will use DataLoaderModule interface to retrieve the needed data in a Tuple format
    """
    
    loader = DataLoaderModule()
    symbols_yfinance = ['SPY']

    indicators = {
        'RSI'               : [4, 14, 20, 25, 30],
        'Stoch'             : [4, 14, 20, 25, 30],
        'ATR'               : [4, 14, 20, 25, 30],
        'MACD'              : [20, 25, 30],
        #'Change'            : [4, 14, 20, 25, 30],
        #'MaxDrawdown'       : [4, 14, 20, 25, 30],
        'CCI'               : [4, 14, 20, 25, 30],
        'Momentum'          : [4, 14, 20, 25, 30],
        'StocksAboveAverage': [4 ,12, 16, 20, 24]
    }

    timeframe        = 'Weekly'
    underlying       = "Close_SPY"

    loader.load_indicators(symbols_yfinance=symbols_yfinance, indicators=indicators, timeframe=timeframe, underlying = underlying)
    loader.get_indicators()
    loader.combine_data()
    data = loader.prepare_data(scale_features=True)

    return data

def get_labels(data: np.array, HMM: bool = True, K_Means: bool = True) -> np.array:

    """
    If both algorithms selected, returns a 2D np array

    Must pass data arguments
    """

    manager = Unsupervised_Learning_Manager(data)

    if HMM and K_Means:
        HMM              = manager.train_hmm(3, True, 5)
        HMM_Predicted    = manager.predict_hmm(HMM)

        KMeans           = manager.train_kmeans(3, True, 5)
        KMeans_Predicted = manager.predict_kmeans(KMeans)

        labels = np.vstack((HMM_Predicted, KMeans_Predicted))
    
    elif HMM:
        HMM              = manager.train_hmm(3)
        HMM_Predicted    = manager.predict_hmm(HMM)

        labels = HMM_Predicted

    elif K_Means:
        KMeans           = manager.train_kmeans(3)
        KMeans_Predicted = manager.predict_kmeans(KMeans)

        labels = KMeans_Predicted
    
    else:
        raise KeyError("[get labels] Please select a valid labelling method")
    
    return labels

def get_maps(features_data: np.array, underlying_data: np.array, manual_confirmation_DTW: bool = True) -> np.array:

    """
    Will return an array full of strin color names (green, yellow, red) which are 
    """

    # Get color maps for each training prediction
    labels = get_labels(features_data)

    color_map_HMM           = classify_states(underlying_data, labels[0], 3, manual_confirmation=manual_confirmation_DTW)
    color_map_KMeans        = classify_states(underlying_data, labels[1], 3, manual_confirmation=manual_confirmation_DTW)

    colors_mapped_HMM       = np.array([color_map_HMM[val] for val in labels[1]])
    colors_mapped_KMeans    = np.array([color_map_KMeans[val] for val in labels[1]])

    maps                    = np.array([color_map_HMM, color_map_KMeans])
    ensembled               = Unsupervised_Learning_Manager().ensemble_predictions(labels, maps)

    color_map_ensembled     = classify_states(underlying_data, ensembled, 3, manual_confirmation=manual_confirmation_DTW)
    colors_mapped_ensembled = np.array([color_map_ensembled[val] for val in ensembled])

    return DataMapped(ensembled, colors_mapped_HMM, colors_mapped_KMeans, colors_mapped_ensembled)


data = get_data()

TrainingDataMapped   = get_maps(data.training_features  , data.training_underlying  , False)
ValidationDataMapped = get_maps(data.validation_features, data.validation_underlying, False)

training_oracle = Oracle_Labeler(TrainingDataMapped.ensembled, data.training_underlying)
training_oracle_labels = training_oracle.modify_states()
training_oracle_labels_mapped = np.array([{0: 'green', 1: 'yellow', 2: 'red'}[val] for val in training_oracle_labels])

validation_oracle = Oracle_Labeler(ValidationDataMapped.ensembled, data.validation_underlying)
validation_oracle_labels = validation_oracle.modify_states()
validation_oracle_labels_mapped = np.array([{0: 'green', 1: 'yellow', 2: 'red'}[val] for val in validation_oracle_labels])

# ---- >>>> Plotting Classified Market <<<< ----

# Training Plot for visual inspection
plotter = Plotting_Module(dates    = data.training_dates, 
                          values   = data.training_underlying, 
                          labels   = ['SPY'],
                          colors   = training_oracle_labels_mapped)
plotter.plot(scatter=False)

# Training Plot for visual inspection
plotter = Plotting_Module(dates    = data.validation_dates, 
                          values   = data.validation_underlying, 
                          labels   = ['SPY'],
                          colors   = validation_oracle_labels_mapped)
plotter.plot(scatter=False)

# ---- >>>> Prediction <<<< ----

# DEFAULT_COLOR_MAP = {0: 'green', 1: 'yellow', 2: 'red'}
supervised_manager = Supervised_Training_Manager(data.training_features, training_oracle_labels, data.validation_features, validation_oracle_labels)

xgb_mod  = supervised_manager.train('xgboost')
#xgb_mod  = supervised_manager.tune_hyperparameters('xgboost')
xgb_pred = [round(x) for x in supervised_manager.predict('xgboost', xgb_mod, data.testing_features)]

rf_mod   = supervised_manager.train('random_forest')
#rf_mod   = supervised_manager.tune_hyperparameters('random_forest')
rf_pred  = [round(x) for x in supervised_manager.predict('random_forest', rf_mod, data.testing_features)]

maps      = np.array([{0: 'green', 1: 'yellow', 2: 'red'}, {0: 'green', 1: 'yellow', 2: 'red'}])

ens_pred  = supervised_manager.ensemble_predictions(np.array([xgb_pred, rf_pred]), maps)
colors_mapped_ens = np.array([{0: 'green', 1: 'yellow', 2: 'red'}[val] for val in ens_pred])

# Training Plot for visual inspection
plotter = Plotting_Module(dates    = data.testing_dates, 
                          values   = data.testing_underlying, 
                          labels   = ['SPY'],
                          colors   = colors_mapped_ens)
plotter.plot(scatter=False)

# ---- >>>> Backtest <<<< ----

bt = Backtest(data.testing_dates, data.testing_underlying, ens_pred,{0: 'green', 1: 'yellow', 2: 'red'})
bt.run_backtest()
bt.plot_balance()


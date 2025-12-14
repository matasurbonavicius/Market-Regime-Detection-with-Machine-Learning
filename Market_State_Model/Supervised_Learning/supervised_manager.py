import pandas as pd
from typing import Any, Union
import numpy as np
from collections import Counter
from typing import Dict

from Supervised_Learning.Random_Forest_model import Random_Forest_Model
from Supervised_Learning.XGBoost_model import XGBoost_Model


class Supervised_Training_Manager:

    def __init__(self, 
        train_data: np.ndarray, 
        train_labels: np.ndarray,
        valid_data: np.ndarray,
        valid_labels: np.ndarray
        ) -> None:
        
        """
        Initialize the Supervised Learning Manager to handle supervised models.
        """

        # Instantiate trainers for both models
        self.xgb_trainer = XGBoost_Model(train_data, train_labels, valid_data, valid_labels)
        self.rf_trainer  = Random_Forest_Model(train_data, train_labels, valid_data, valid_labels)

    def train(self, model_type: str, params: dict = None) -> Any:
        
        """
        Train the specified supervised model. 
        The model can be 'xgboost' or 'random_forest'.
        """

        if model_type == "xgboost":
            return self.xgb_trainer.train_model(params)
        elif model_type == "random_forest":
            return self.rf_trainer.train_model(params)
        else:
            raise ValueError("Model type must be either 'xgboost' or 'random_forest'.")

    def tune_hyperparameters(self, model_type: str) -> Any:
        
        """
        Tune the hyperparameters of the specified supervised model.
        """

        if model_type == "xgboost":
            return self.xgb_trainer.tune_hyperparameters()
        elif model_type == "random_forest":
            return self.rf_trainer.tune_hyperparameters()
        else:
            raise ValueError("Model type must be either 'xgboost' or 'random_forest'.")

    def predict(self, model_type: str, model: Any, data: Union[pd.DataFrame, pd.Series, None] = None) -> np.array:
        
        """
        Predict using the trained supervised model.
        """

        if model_type == "xgboost":
            return self.xgb_trainer.predict(model, data)
        elif model_type == "random_forest":
            return self.rf_trainer.predict(model, data)
        else:
            raise ValueError("Model type must be either 'xgboost' or 'random_forest'.")

    def visualize_model(self, model_type: str):
        
        """
        Visualize the trained model's characteristics.
        """

        if model_type == "xgboost":
            return self.xgb_trainer.plot_trained_model()
        elif model_type == "random_forest":
            return self.rf_trainer.plot_feature_importance()
        else:
            raise ValueError("Model type must be either 'xgboost' or 'random_forest'.")
    
    def ensemble_predictions(self, predictions: np.ndarray, color_maps: np.ndarray[Dict[int, str]]) -> np.ndarray:

        """
        Ensembling predictions to have a unified array

        Rules:
            - Voting. Majority always wins
            - If all votes are different, use the best prediction color from green to red
                * This is due to the positive long-term growth of the stock market

        Arguments:
            predictions: np.array: an array consisting of prediction values. Can (should) be multi dimensional
                        where each array corresponds to a different prediction series

            color_maps: np.ndarray[Dict[int, str]] or List[Dict[int, str]]: an array consisting of dictionaries that map
                        the predicted values with the color code. example:
                        * [{0: 'green', 1:'yellow', 2:'red'},
                           {0: 'green', 1:'red', 2:'yellow'}]
        """
        
        DEFAULT_COLOR_MAP = {0: 'green', 1: 'yellow', 2: 'red'}
        DEFAULT_COLOR_VALUES = list(DEFAULT_COLOR_MAP.values())

        # Convert numeric predictions to colors using the respective color maps
        color_predictions = []
        for i, preds in enumerate(predictions):

            color_preds = []
            for pred in preds:
                
                color = color_maps[i].get(pred, 'unknown')
                color_preds.append(color)
            
            color_predictions.append(color_preds)
        
        def best_color(color_preds):
            # Count occurrences
            counts = Counter(color_preds)
            
            # MAJORITY VOTING (ALL IN)
            # If all predictions are the same, return that value
            if len(counts) == 1:
                return color_preds[0]
            
            # ALL DIFFERENT? - CHOOSE THE BEST
            if len(counts) == len(color_preds):
                for color in DEFAULT_COLOR_VALUES:
                    if color in color_preds:
                        return color
            
            # MAJORITY VOTING
            # Returns the most common (the mode, statistically)
            return counts.most_common(1)[0][0]
        
        # Transpose to group predictions from different algorithms for the same input
        transposed_colors = list(zip(*color_predictions))
        
        # Apply the best_color function
        ensembled_colors = [best_color(color_group) for color_group in transposed_colors]
        
        # Convert ensembled colors back to numerical values; map them to DEFAULT_COLOR_VALUES
        ensembled_numbers = [DEFAULT_COLOR_VALUES.index(color) for color in ensembled_colors]

        return np.array(ensembled_numbers)

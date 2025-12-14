from sklearn.cluster import KMeans
from hmmlearn import hmm as HMM
from collections import Counter
from typing import Dict
import logging
import numpy as np

from Unsupervised_Learning.HMM_model import HMM_Model
from Unsupervised_Learning.K_means_model import K_Means_Model

class Unsupervised_Learning_Manager:

    def __init__(self, data: np.ndarray = None) -> None:
        
        if data is not None:
            self.data = data
            self.kmeans_trainer = K_Means_Model(data)
            self.hmm_trainer = HMM_Model(data)

    def train_kmeans(self, n_clusters: int, multiple_times: bool = False, n_train: int = 1) -> object:
        """Train the KMeans model."""
        if multiple_times:
            return self.kmeans_trainer.train_n_models(n_train, n_clusters)
        else:
            return self.kmeans_trainer.train_model(n_clusters)

    def train_hmm(self, n_components: int, multiple_times: bool = False, n_train: int = 1) -> object:
        """Train the HMM model."""
        if multiple_times:
            return self.hmm_trainer.train_n_models(n_train, n_components)
        else:
            return self.hmm_trainer.train_model(n_components)

    def predict_kmeans(self, model: KMeans) -> np.array:
        """Predict using KMeans model."""
        return self.kmeans_trainer.predict(model, self.data)

    def predict_hmm(self, model: HMM.GaussianHMM) -> np.array:
        """Predict using HMM model."""
        return self.hmm_trainer.predict(model, self.data)

    def ensemble_predictions(self, predictions: np.ndarray, color_maps: np.ndarray[Dict[int, str]]) -> np.ndarray:

        """
        Ensembling predictions to have a unified array

        Rules:
            - Voting. Majority always wins
            - If all votes are different, use the best prediction color from green to red
                * This is due to the positive long-term growth of the stock market

        Arguments:
            predictions: np.array: an array consisting of prediction values. Can be multi dimensional
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
            if len(counts) == len(DEFAULT_COLOR_MAP):
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
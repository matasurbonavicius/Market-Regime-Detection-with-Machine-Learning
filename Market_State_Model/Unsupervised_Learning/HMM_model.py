from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm as HMM
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import logging


class HMM_Model:
    """Class for training Hidden Markov Models."""

    """
    # Sample code usage
    data = np.random.randn(100, 1)
    trainer = HMM_Model(data)
    model = trainer.train_model(2)
    predictions = trainer.predict(model, data)
    logging.info("Model Training and Prediction Completed.")
    """

    def __init__(self, 
                 data: np.ndarray, 
                 means: np.ndarray = None, 
                 covars: np.ndarray = None, 
                 startprob: np.ndarray = None, 
                 transmat: np.ndarray = None) -> None:
        
        """
        Initialize the HMM_Model class.

        Parameters:
        - data (np.ndarray): Data to train on, must be scaled. 
        - means (np.ndarray): Means for the HMM.
        - covars (np.ndarray): Covariance matrices for the HMM.
        - startprob (np.ndarray): Starting probabilities for the HMM.
        - transmat (np.ndarray): Transition matrix for the HMM.
        """

        self.means = means
        self.covars = covars
        self.startprob = startprob
        self.transmat = transmat
        self.data = data

    def train_model(self, n_components: int) -> HMM.GaussianHMM:
        
        """
        Train a GaussianHMM model.

        Parameters:
        - n_components (int): Number of states in the HMM.

        Returns:
        - HMM.GaussianHMM: Trained model.
        """

        if self.means is None:
            model = HMM.GaussianHMM(n_components=n_components, 
                                    covariance_type="full", 
                                    n_iter=1000)
        else:
            model = HMM.GaussianHMM(n_components=n_components, 
                                    covariance_type="full", 
                                    n_iter=1000, 
                                    init_params="")
            try:
                model.startprob_ = self.startprob
                model.transmat_ = self.transmat
                model.means_ = self.means
                model.covars_ = self.covars
            except: 
                try:
                    model.startprob_ = self.startprob
                    model.transmat_ = self.transmat
                    model.means_ = self.means.diagonal(axis1=1, axis2=2)
                    model.covars_ = self.covars.diagonal(axis1=1, axis2=2)
                except:
                    logging.error("Model train not successful. Check parameters.")
                    return

        model = model.fit(self.data)
        return model

    def train_n_models(self, n_train, n_components) -> None:
        
        """
        Train the HMM model multiple times and retain the best model based on score.
        """

        best_score = float('-inf')

        if self.means is None:

            for i in range(n_train):
                
                model = HMM.GaussianHMM(
                    n_components=n_components, 
                    covariance_type="full", 
                    n_iter=1000
                    )
                
                model.fit(self.data)
                score = model.score(self.data)
                
                if score > best_score:
                    best_score = score
                    self.best_model = model
                logging.info(f"Training iteration {i + 1}/{n_train} - Score: {score}")
            logging.info(f"Best model chosen with Score: {self.best_model.score(self.data)}")

        return self.best_model

    def predict(self, model: HMM.GaussianHMM, data: np.ndarray) -> np.ndarray:
        
        """
        Predict states using the trained model.

        Parameters:
        - model (HMM.GaussianHMM): Trained HMM model.
        - data (np.ndarray): Data for prediction.

        Returns:
        - np.ndarray: Predicted states.
        """

        return model.predict(data)


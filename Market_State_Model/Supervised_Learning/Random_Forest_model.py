from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class Random_Forest_Model():
    """
    A class to handle Random Forest training, hyperparameter tuning, 
    prediction, and visualization.

    Attributes:
    - feature (pd.DataFrame): Features used for training and testing.
    - target (pd.Series): Target values.
    """

    def __init__(self, 
        train_data: np.ndarray, 
        train_labels: np.ndarray,
        valid_data: np.ndarray,
        valid_labels: np.ndarray
        ) -> None:
        """
        Initialize RandomForest_Trainer instance and split data into 
        training and test sets.
        """

        self.X_train = train_data
        self.y_train = train_labels
        self.X_valid = valid_data
        self.y_valid = valid_labels

    def train_model(self, params: dict = None) -> RandomForestRegressor:
        """
        Train the Random Forest model.
        """

        # Default Random Forest parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': None
        }

        # If custom parameters are provided, update defaults with them
        if params:
            default_params.update(params)

        logging.info("Starting model training.")

        self.rf = RandomForestRegressor(**default_params)
        self.rf.fit(self.X_train, self.y_train)
        
        logging.info("Model training complete.")
        
        return self.rf

    def tune_hyperparameters(self) -> RandomForestRegressor:
        
        """
        Tune the model hyperparameters using RandomizedSearchCV.

        Returns:
        - RandomForestRegressor: Best model found.
        """

        # Define hyperparameter search space
        param_dist = {
            'n_estimators': [10, 50, 100, 200, 500, 1000],
            'max_depth': [1, 5,  10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [1, 100],
            'bootstrap': [True, False]
        }

        logging.info("Starting hyperparameter tuning.")

        try:
            combined_data = np.concatenate([np.array(self.X_train), np.array(self.X_valid)], axis=0)
            combined_labels = np.concatenate([np.array(self.y_train).ravel(), np.array(self.y_valid).ravel()])
        except:
            raise ValueError("[tune_hyperparameters] Failed to combine data & labels")

        # Define the split between training and validation
        test_fold = [-1] * len(self.X_train) + [0] * len(self.X_valid)
        ps = PredefinedSplit(test_fold)

        rf_model = RandomForestRegressor()
        rs = RandomizedSearchCV(
            rf_model, 
            param_dist, 
            n_iter=100, 
            scoring="neg_mean_squared_error", 
            verbose=1, 
            cv=ps, 
            n_jobs=-1
            )
        
        rs.fit(combined_data, combined_labels)
        self.bst = rs.best_estimator_
        
        logging.info("Hyperparameter tuning complete.")
        
        return self.bst
    
    def predict(self, model: RandomForestRegressor, data: Union[pd.DataFrame, pd.Series, None] = None) -> np.array:
        """
        Predict target values using the trained model.
        """
        
        if data is None:
            data = self.X_test
        
        # Convert pd.Series to pd.DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame().transpose()

        return model.predict(data)

    def plot_feature_importance(self) -> None:
        """
        Visualize the trained model's feature importance.
        """

        logging.info("Generating Feature importances visualization.")
        
        feature_importance = self.rf.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        try:
            plt.barh(np.array(self.X_train.columns)[sorted_idx], feature_importance[sorted_idx])
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.title('Feature Importance')
            
            plt.tight_layout()
            plt.show()
        except:
            try:
                plt.barh(np.array(self.X_train)[sorted_idx], feature_importance[sorted_idx])
                plt.xlabel('Importance')
                plt.ylabel('Features')
                plt.title('Feature Importance')
                
                plt.tight_layout()
                plt.show()
            except:
                pass

        logging.info("Feature importance visualization displayed.")

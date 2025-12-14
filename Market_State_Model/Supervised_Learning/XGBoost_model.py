from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
from typing import Union
import matplotlib.gridspec as gridspec
from typing import Any
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s]: %(message)s"
    )

# ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class XGBoost_Model():
    
    """
    A class to handle XGBoost training, hyperparameter tuning, 
    prediction, and visualization.

    Attributes:
    - feature (pd.DataFrame): Features used for training and testing.
    - target (pd.Series): Target values.
    """

    def __init__(self, 
                 train_data: np.ndarray, 
                 train_labels: np.ndarray,
                 valid_data: np.ndarray,
                 valid_labels: np.ndarray) -> None:
    
        
        """
        Initialize XGB_Trainer instance and split data into training 
        and test sets.

        Parameters:
        - feature (pd.DataFrame): Features.
        - target (pd.Series): Target values.
        - test_size (float, optional): Fraction of dataset to use as test set. 
            Defaults to 0.2.
        - random_state (int, optional): Seed value. Defaults to 42.

        Returns:
        - None
        """

        self.X_train = train_data
        self.y_train = train_labels
        self.X_valid = valid_data
        self.y_valid = valid_labels
        
        self.dtrain = xgb.DMatrix(data = self.X_train, label = self.y_train)
        self.dvalid = xgb.DMatrix(data = self.X_valid, label = self.y_valid)

        self.evals_result = {}

    def train_model(self, params: dict = None) -> xgb.Booster:
        
        """
        Train the XGBoost model.

        Parameters:
        - params (dict, optional): Parameters to use in training. 
            Overrides defaults if provided.

        Returns:
        - xgb.Booster: Trained model.
        """

        # Default XGBoost parameters
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'eta': 0.3
        }

        # If custom parameters are provided, update defaults with them
        if params:
            default_params.update(params)

        logging.info("Starting model training.")

        # Train the model
        num_rounds = 10
        watchlist = [(self.dtrain, 'train'), (self.dvalid, 'valid')]
        self.bst = xgb.train(
            default_params, 
            self.dtrain, 
            num_rounds, 
            evals=watchlist, 
            evals_result=self.evals_result
            )
        
        logging.info("Model training complete.")
        
        return self.bst

    def tune_hyperparameters(self) -> xgb.XGBRegressor:
        
        """
        Tune the model hyperparameters using RandomizedSearchCV.

        Returns:
        - xgb.XGBRegressor: Best model found.
        """

        # Define hyperparameter search space
        param_dist = {
            'eta': [0.01, 0.05, 0.1, 0.2, 0.3],
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.25, 0.5, 1.0],
            'lambda': [0, 0.1, 1.0],
            'alpha': [0, 0.1, 1.0]
        }

        logging.info("Starting hyperparameter tuning.")

        combined_data = np.concatenate([self.X_train, self.X_valid], axis=0)
        combined_labels = np.concatenate([self.y_train, self.y_valid])

        # Define the split between training and validation
        test_fold = [-1] * len(self.X_train) + [0] * len(self.X_valid)
        ps = PredefinedSplit(test_fold)

        xgb_model = xgb.XGBRegressor()
        rs = RandomizedSearchCV(
            xgb_model, 
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
    
    def predict(self, model: Any, data: Union[pd.DataFrame, pd.Series, None] = None) -> np.array:
        """
        Predict target values using the trained model.

        Parameters:
        - data (Union[pd.DataFrame, pd.Series, None], optional): Data for which predictions are made. 
            If not provided, uses test set.

        Returns:
        - np.array: Predicted values.
        """
        
        if data is None:
            data = self.X_test

        # Convert pd.Series to pd.DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame().transpose()
        
        # First try prediction with data as argument, 
        # if failed, then try converting it to DMatrix and running
        try:
                return model.predict(data)
        except:
            try:
                if not isinstance(data, xgb.DMatrix):
                    data_dmatrix = xgb.DMatrix(data)
                else:
                    data_dmatrix = data

                return model.predict(data_dmatrix)
            
            except Exception:
                raise ValueError("[predict] Prediction Failed")

    def plot_trained_model(self) -> None:
        """
        Visualize the trained model's feature importance and RMSE over iterations, 
        tree structure.

        Returns:
        - None. Plots are displayed.
        """

        logging.info("Generating tree structure visualization.")
        
        fig, ax = plt.subplots(figsize=(15, 10))
        xgb.plot_tree(self.bst, num_trees=2, ax=ax)
        ax.set_title("Tree Structure")
        
        plt.tight_layout()
        plt.show()

        logging.info("Tree structure visualization displayed.")
        logging.info("Generating Feature importances & RMSE visualization.")

        # Create a gridspec layout for the plots
        fig = plt.figure(figsize=(10, 10))
        spec = gridspec.GridSpec(nrows=2, ncols=1)

        # Feature Importance Plot
        ax1 = fig.add_subplot(spec[0, 0])
        xgb.plot_importance(self.bst, importance_type='weight', show_values=True, ax=ax1)
        ax1.set_title("Feature Importance")
        logging.info("Feature importance visualization generated.")

        # Training and Evaluation RMSE Over Iterations
        evals_result = self.evals_result
        train_rmse = evals_result['train']['rmse']
        eval_rmse = evals_result['valid']['rmse']

        ax2 = fig.add_subplot(spec[1, 0])
        ax2.plot(range(len(train_rmse)), train_rmse, '-o', label='Train RMSE')
        ax2.plot(range(len(eval_rmse)), eval_rmse, '-o', label='Valid RMSE')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Training and Evaluation RMSE Over Iterations')
        ax2.legend()
        ax2.grid(True)

        logging.info("Training and Evaluation RMSE visualization generated.")

        plt.tight_layout()
        plt.show()

        logging.info("Model visualizations for features and RMSE displayed.")
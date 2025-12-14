import numpy as np
from sklearn.cluster import KMeans
import logging


class K_Means_Model:
    """Class for training K-Means Clustering."""

    """
    # Sample code usage
    data = np.random.randn(100, 2)
    trainer = KMeansTrainer(data)
    model = trainer.train_model(3)
    predictions = trainer.predict(model, data)
    logging.info("Model Training and Prediction Completed.")
    """

    def __init__(self, 
                 data: np.ndarray, 
                 init: str = 'k-means++',
                 n_init: int = 10,
                 max_iter: int = 300) -> None:
        
        """
        Initialize the KMeansTrainer class.

        Parameters:
        - data (np.ndarray): Data to train on.
        - init (str): Method for initialization.
        - n_init (int): Number of time the k-means algorithm will be run with different centroid seeds.
        - max_iter (int): Maximum number of iterations of the k-means algorithm for a single run.
        """

        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.data = data

    def train_model(self, n_clusters: int) -> KMeans:
        
        """
        Train a KMeans model.

        Parameters:
        - n_clusters (int): The number of clusters to form as well as the number of centroids to generate.

        Returns:
        - KMeans: Trained model.
        """

        model = KMeans(n_clusters=n_clusters, 
                       init=self.init, 
                       n_init=self.n_init, 
                       max_iter=self.max_iter)
        
        model.fit(self.data)

        return model

    def train_n_models(self, n_train, n_clusters) -> None:
        
        """
        Train the KMeans model multiple times and retain the best model based on inertia.
        """

        best_inertia = float('inf')

        for i in range(n_train):
                
            model = KMeans(n_clusters=n_clusters, 
                           init=self.init, 
                           n_init=self.n_init, 
                           max_iter=self.max_iter)
                
            model.fit(self.data)
            inertia = model.inertia_
                
            if inertia < best_inertia:
                best_inertia = inertia
                self.best_model = model
            logging.info(f"Training iteration {i + 1}/{n_train} - Inertia: {inertia}")
        logging.info(f"Best model chosen with Inertia: {self.best_model.inertia_}")

        return self.best_model

    def predict(self, model: KMeans, data: np.ndarray) -> np.ndarray:
        
        """
        Predict the closest cluster each sample in the data belongs to.

        Parameters:
        - model (KMeans): Trained KMeans model.
        - data (np.ndarray): Data for prediction.

        Returns:
        - np.ndarray: Predicted clusters.
        """

        return model.predict(data)


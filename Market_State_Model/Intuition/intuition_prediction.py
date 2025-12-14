import pandas as pd
import numpy as np

from Supervised_Learning.supervised_manager import Supervised_Training_Manager
from Supervised_Learning.XGBoost_model import XGBoost_Model
from Supervised_Learning.Random_Forest_model import Random_Forest_Model
import xgboost as xgb

training_features = pd.DataFrame({
    'engine':   [5  ,2  ,1.7,1.9,5.5, 3.5, 8 , 7 , 2 , 6 ],
    'wheels':   [4  ,4  ,4  ,2  ,4  , 4  , 6 , 6 , 2 , 4 ],
    'top_sp':   [310,180,150,200,280,290 ,80 ,120,270,90 ],
    'priceK':   [300,20 ,16 ,9  ,55 ,230 ,700,300,15 ,650]
})

training_labels = pd.DataFrame({
    'supercar or truck': [1,  0,  0,  0,  0,  1,  2,  2,  0,  2]
})

validation_features = pd.DataFrame({
    'engine':   [4  ,5  ,6  ],
    'wheels':   [4  ,4  ,4  ],
    'top_sp':   [290,130,70 ],
    'priceK':   [270,100,400]
})

validation_labels = pd.DataFrame({
    'supercar or truck': [1, 0, 2]
})

# --------------------------------------

testing_features = pd.DataFrame({
    'engine':   [1.9,4  ,3  , 8  ],
    'wheels':   [2  ,4  ,4  , 6  ],
    'top_sp':   [180,330,260, 90 ],
    'priceK':   [25 ,350,65 , 650]
})

supervised_manager = Supervised_Training_Manager(training_features, training_labels, validation_features, validation_labels)

xgb_mod  = supervised_manager.train('xgboost')
xgb_mod  = supervised_manager.tune_hyperparameters('xgboost')
xgb_pred = [round(x) for x in supervised_manager.predict('xgboost', xgb_mod, testing_features)]
supervised_manager.visualize_model('xgboost')

rf_mod  = supervised_manager.train('random_forest')
rf_mod  = supervised_manager.tune_hyperparameters('random_forest')
rf_pred = [round(x) for x in supervised_manager.predict('random_forest', rf_mod, testing_features)]
supervised_manager.visualize_model('random_forest')

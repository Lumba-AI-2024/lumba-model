import subprocess

# Upgrade TensorFlow
subprocess.run(["pip", "install", "--upgrade", "tensorflow"])

# Upgrade Keras
subprocess.run(["pip", "install", "--upgrade", "keras"])

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV,train_test_split, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas.core.frame import DataFrame

from typing import Any, Optional, Union, List

class LumbaNeuralNetworkRegression:
    model: KerasClassifier

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe = dataframe

    def train_model(self, target_column_name: str) -> dict:
        # if self.dataframe[target_column_name].dtype not in ["int64", "float64"]:
        #     return {
        #         'error': 'Kolom yang boleh dipilih hanyalah kolom dengan data numerik saja. Silakan pilih kolom yang benar.'
        #     }
        
        # x = None
        # if type(train_column_name) == str:
        #     if self.dataframe[train_column_name].dtype not in ["int64", "float64"]:
        #         return {
        #             'error': 'Kolom yang boleh dipilih hanyalah kolom dengan data numerik saja. Silakan pilih kolom yang benar.'
        #         }
        #     x = self.dataframe[train_column_name].to_numpy().reshape(-1, 1)
        
        # elif type(train_column_name) == list:
        #     for col in train_column_name:
        #         if self.dataframe[col].dtype not in ["int64", "float64"]:
        #             return {
        #                 'error': 'Kolom yang boleh dipilih hanyalah kolom dengan data numerik saja. Silakan pilih kolom yang benar.'
        #             }

        #     x = self.dataframe[train_column_name].to_numpy()

        # y = self.dataframe[target_column_name].to_numpy().reshape(-1, 1)

        X = self.dataframe.drop(columns=[target_column_name])
        y = self.dataframe[target_column_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Set seed for NumPy
        np.random.seed(42)

        # Set seed for TensorFlow
        tf.random.set_seed(42)
        
        # Define a function to create model
        def create_model(optimizer='adam', activation='relu', units1=64, units2=32):
            model = Sequential([
                Dense(units1, activation=activation, input_shape=(X.shape[1],)),
                Dense(units2, activation=activation),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            return model        

        # Wrap the Keras model in a KerasClassifier
        kc = KerasClassifier(build_fn=create_model, verbose=0)

        # Define the grid search parameters
        param_grid = {
            'model__optimizer': ['adam', 'rmsprop'],
            'model__activation': ['relu', 'sigmoid'],
            'model__units1': [32, 64, 128],
            'model__units2': [16, 32, 64],
            'epochs': [10, 20, 30, 40, 50]  # Adjust the values as needed
        }

        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Perform grid search
        grid = GridSearchCV(estimator=kc, param_grid=param_grid, cv=outer_cv)
        grid_result = grid.fit(X_train, y_train)  # Assuming X and y are your feature matrix and target vector
        
        best_hyperparams = grid_result.best_params_

        # Evaluate the best model and count accuracy
        best_model = grid_result.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)


        self.model = kc

        return {
            'model': kc,
            'best_hyperparams': best_hyperparams,
            'accuracy_score': f'{acc*100:.4f}'
        }
    
    def get_model(self) -> Optional[KerasClassifier]:
        try:
            return self.model
        except AttributeError:
            return None

    # def predict(self, data_target: Any) -> Any:
    #     lr = self.get_model()
    #     y_pred = lr.predict(data_target)

    #     return y_pred
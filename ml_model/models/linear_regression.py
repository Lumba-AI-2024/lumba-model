from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV,train_test_split, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pandas as pd

from pandas.core.frame import DataFrame

from typing import Any, Optional, Union, List

class LumbaLinearRegression:
    model: LinearRegression

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
        
        lr = LinearRegression()

        # Define the grid search parameters
        param_grid = {
            'fit_intercept': [True, False],  # Whether to calculate the intercept for this model
            'copy_X': [True, False],  # Whether to make a copy of X before fitting the model
            'n_jobs': [-1],  # Number of jobs to run in parallel. -1 means using all processors.
            'positive' : [True, False]
        }

        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Perform grid search
        grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=outer_cv)
        grid_result = grid.fit(X_train, y_train)  # Assuming X and y are your feature matrix and target vector
        
        best_hyperparams = grid_result.best_params_

        # Evaluate the best model and count accuracy
        best_model = grid_result.best_estimator_
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_true=y_test, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        mse = mean_squared_error(y_true=y_test, y_pred=y_pred)

        self.model = best_model

        X_test_df = pd.DataFrame(X_test, columns=self.dataframe.drop(columns=[target_column_name]).columns)
        X_train_df = pd.DataFrame(X_train, columns=self.dataframe.drop(columns=[target_column_name]).columns)
        
        return {
            'model': best_model,
            'X_train': X_train_df,
            'X_test': X_test_df,
            'best_hyperparams': best_hyperparams,
            'mean_absolute_error': f'{mae:.4f}',
            'mean_squared_error': f'{mse:.4f}',
            'r2_score': f'{r2:.4f}'
        }
    
    def get_model(self) -> Optional[LinearRegression]:
        try:
            return self.model
        except AttributeError:
            return None

    # def predict(self, data_target: Any) -> Any:
    #     lr = self.get_model()
    #     y_pred = lr.predict(data_target)

    #     return y_pred
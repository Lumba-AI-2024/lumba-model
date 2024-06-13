from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,train_test_split, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import pandas as pd
import time
from pandas.core.frame import DataFrame
from pandas import Series

from typing import Any, Optional, Union, List

class LumbaXGBoostClassifier:
    model: XGBClassifier

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe = dataframe

    def train_model(self, target_column_name: str) -> dict:
        
        X = self.dataframe.drop(columns=[target_column_name])
        y = self.dataframe[target_column_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'max_depth': [2, 4, 8, 10],
            'min_child_weight': [1, 5, 10],
            'booster':['gbtree','gblinear'],
            'base_score':[0.25,0.5,0.75,1]
        }

        xg = XGBClassifier(random_state=42, nthread=4)

        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Perform grid search
        grid = GridSearchCV(estimator=xg, param_grid=param_grid, cv=outer_cv, scoring='accuracy', verbose=1)
        grid_result = grid.fit(X_train, y_train)  # Assuming X and y are your feature matrix and target vector
        
        best_hyperparams = grid_result.best_params_

        # Evaluate the best model and count accuracy
        best_model = grid_result.best_estimator_
        start_time = time.time()
        y_pred = best_model.predict(X_test)
        end_time = time.time()
        num_classes = len(pd.unique(y))
        if num_classes > 2:
            average_method = 'macro'
        else:
            average_method = 'binary'

        acc = accuracy_score(y_true=y_test, y_pred=y_pred)
        recall = recall_score(y_true=y_test, y_pred=y_pred, average=average_method)
        precision = precision_score(y_true=y_test, y_pred=y_pred, average=average_method)
        f1 = f1_score(y_true=y_test, y_pred=y_pred, average=average_method)
        elapsed_time = end_time - start_time

        self.model = best_model

        X_test_df = pd.DataFrame(X_test, columns=self.dataframe.drop(columns=[target_column_name]).columns)
        X_train_df = pd.DataFrame(X_train, columns=self.dataframe.drop(columns=[target_column_name]).columns)
        
        return {
            'model': best_model,
            'X_train': X_train_df,
            'X_test': X_test_df,
            'best_hyperparams': best_hyperparams,
            'accuracy_score': acc,
            'recall_score': recall,
            'precision_score': precision,
            'f1_score': f1,
            'best_hyperparams': best_hyperparams,
            'time': elapsed_time,
        }
    
    def get_model(self) -> Optional[XGBClassifier]:
        try:
            return self.model
        except AttributeError:
            return None

    # def predict(self, data_target: Any) -> Any:
    #     lr = self.get_model()
    #     y_pred = lr.predict(data_target)

    #     return y_pred
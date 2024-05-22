from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,train_test_split, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

        xg = XGBClassifier(random_state=42)

        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Perform grid search
        grid = GridSearchCV(estimator=xg, param_grid=param_grid, cv=outer_cv, scoring='accuracy', verbose=1)
        grid_result = grid.fit(X_train, y_train)  # Assuming X and y are your feature matrix and target vector
        
        best_hyperparams = grid_result.best_params_

        # Evaluate the best model and count accuracy
        best_model = grid_result.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)

        self.model = xg

        return {
            'model': xg,
            'best_hyperparams': best_hyperparams,
            'accuracy_score': f'{acc*100:.4f}'
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
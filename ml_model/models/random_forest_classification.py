from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split, KFold
from sklearn.metrics import accuracy_score

from pandas.core.frame import DataFrame
from pandas import Series
from typing import Any, List, Optional

class LumbaRandomForestClassifier:
    model: RandomForestClassifier

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe = dataframe

    def train_model(self, target_column_name: str) -> dict:
        
        X = self.dataframe.drop(columns=[target_column_name])
        y = self.dataframe[target_column_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestClassifier(random_state = 42)

        param_grid = {
            'n_estimators': [50, 100, 150],  # Number of trees in the forest
            'criterion': ['gini', 'entropy'],
            'max_depth': [2, 4, 8, 10],
        }

        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=outer_cv, return_train_score=True)
        grid_result = grid.fit(X_train, y_train)

        best_hyperparams = grid_result.best_params_

        # Evaluate the best model and count accuracy
        best_model = grid_result.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)

        self.model = best_model

        return {
            'model': best_model,
            'best_hyperparams': best_hyperparams,
            'accuracy_score': f'{acc*100:.4f}'
        }

    def get_model(self) -> Optional[RandomForestClassifier]:
        try:
            return self.model
        except AttributeError:
            return None

    # def predict(self, data_target: Any) -> Any:
    #     dt = self.get_model()
    #     y_pred = dt.predict(data_target)

    #     return y_pred
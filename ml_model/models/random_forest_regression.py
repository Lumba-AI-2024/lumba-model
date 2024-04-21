from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from pandas.core.frame import DataFrame

from typing import Any, List, Optional

class LumbaRandomForestClassifier:
    model: RandomForestRegressor

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe = dataframe

    def train_model(self, target_column_name: str) -> dict:
        
        # # check if the columns selected are valid for Decision Tree process
        # for col in x.columns:
        #     if y.dtype not in ["int64", "float64"] or x[col].dtype not in ["int64", "float64"]:
        #         return {
        #             'error': 'Kolom yang boleh dipilih hanyalah kolom dengan data numerik saja. Silakan pilih kolom yang benar atau gunakan encoding pada data categorical.'
        #         }
        
        X = self.dataframe.drop(columns=[target_column_name])
        y = self.dataframe[target_column_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestRegressor(random_state = 42)

        param_grid = {
            'n_estimators': [50, 100, 150],  # Number of trees in the forest
            'criterion': ['squared_error', 'poisson', 'friedman_mse', 'absolute_error'],
            'max_depth': [2, 4, 8, 10],
        }

        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=outer_cv, return_train_score=True)
        grid_result = grid.fit(X_train, y_train)

        best_hyperparams = grid_result.best_params_

        # Evaluate the best model and count accuracy
        best_model = grid_result.best_estimator_
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_true=y_test, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        mse = mean_squared_error(y_true=y_test, y_pred=y_pred)

        self.model = rf

        return {
            'model': rf,
            'best_hyperparams': best_hyperparams,
            'mean_absolute_error': f'{mae:.4f}',
            'mean_squared_error': f'{mse:.4f}',
            'r2_score': f'{r2:.4f}'
        }

    def get_model(self) -> Optional[RandomForestRegressor]:
        try:
            return self.model
        except AttributeError:
            return None

    # def predict(self, data_target: Any) -> Any:
    #     dt = self.get_model()
    #     y_pred = dt.predict(data_target)

    #     return y_pred
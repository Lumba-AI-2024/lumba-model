# https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import optuna

from pandas.core.frame import DataFrame

from typing import Optional

class LumbaDBScan:
    model: DBSCAN

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe = dataframe

    def train_model(self) -> dict:
        X = self.dataframe
        
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
        # # check if the columns selected are valid for K-Means process
        # for col in x.columns:
        #     if x[col].dtype not in ["int64", "float64"]:
        #         return {
        #             'error': 'Kolom yang boleh dipilih hanyalah kolom dengan data numerik saja. Silakan pilih kolom yang benar atau gunakan encoding pada data categorical.'
        #         }
        def dbscan_objective(trial):
            eps = trial.suggest_uniform('eps', 0.1, 2.0)
            min_samples = trial.suggest_int('min_samples', 2, 20)
            
            # Valid metrics for KD-tree algorithm
            valid_metrics_kd_tree = ['euclidean', 'manhattan']
            metric = trial.suggest_categorical('metric', valid_metrics_kd_tree)
            
            algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm)
            dbscan.fit(X)
            
            if len(set(dbscan.labels_)) > 1:  # Silhouette score requires at least 2 clusters
                silhouette_avg = silhouette_score(X, dbscan.labels_)
            else:
                silhouette_avg = -1  # Return a negative score if only one cluster is found
            
            return silhouette_avg
        
                # Perform hyperparameter optimization for DBSCAN
        dbscan_study = optuna.create_study(direction='maximize')
        dbscan_study.optimize(dbscan_objective, n_trials=100)

        # Get optimal hyperparameters from the study results
        optimal_dbscan_eps = dbscan_study.best_params['eps']
        optimal_dbscan_min_samples = dbscan_study.best_params['min_samples']
        optimal_dbscan_metric = dbscan_study.best_params['metric']
        optimal_dbscan_algorithm = dbscan_study.best_params['algorithm']

        optimal_dbscan = DBSCAN(eps=optimal_dbscan_eps, min_samples=optimal_dbscan_min_samples, metric=optimal_dbscan_metric, algorithm=optimal_dbscan_algorithm)
        optimal_dbscan.fit(X)

        # predicted cluster labels
        dbscan_cluster_labels = optimal_dbscan.labels_

        # silhouette score
        silhouette = silhouette_score(X, dbscan_cluster_labels)

        self.model = optimal_dbscan

        return {
            'model': optimal_dbscan,
            'cluster_labels': dbscan_cluster_labels,
            'silhouette_score': f'{silhouette:.4f}'
        }

    def get_model(self) -> Optional[DBSCAN]:
        try:
            return self.model
        except AttributeError:
            return None

    
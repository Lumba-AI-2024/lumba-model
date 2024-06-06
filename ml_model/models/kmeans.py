# https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import optuna

from pandas.core.frame import DataFrame

from typing import Any, Optional, List
from sklearn.ensemble import RandomForestClassifier


class LumbaKMeans:
    model: KMeans

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
        def kmeans_objective(trial):
            n_clusters = trial.suggest_int('n_clusters', 2, 20)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')
            kmeans.fit(X)
            
            silhouette_avg = silhouette_score(X, kmeans.labels_)
            
            return silhouette_avg
        
        kmeans_study = optuna.create_study(direction='maximize')
        kmeans_study.optimize(kmeans_objective, n_trials=100)

        optimal_kmeans_n_clusters = kmeans_study.best_params['n_clusters']

        k = optimal_kmeans_n_clusters

        km_model = KMeans(n_clusters=k, random_state=42, init='k-means++')
        km_model.fit_predict(X)

        # predicted cluster labels
        kmeans_cluster_labels = km_model.labels_

        # silhouette score
        silhouette = silhouette_score(X, kmeans_cluster_labels)

        self.model = km_model
        
        shap_model=RandomForestClassifier()
        shap_model.fit(self.dataframe,kmeans_cluster_labels)
        
        best_hyperparams = kmeans_study.best_params

        return {
            'model': km_model,
            'shap_model': shap_model,
            'silhouette_score': f'{silhouette:.4f}',
            'best_hyperparams': best_hyperparams
        }

    def get_model(self) -> Optional[KMeans]:
        try:
            return self.model
        except AttributeError:
            return None

    
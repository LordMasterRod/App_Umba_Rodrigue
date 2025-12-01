# clustering.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

class DynamicClusteringGeneralized:
    # ici tu peux copier ta version précédente avec fit, assign, prototypes etc.
    pass

class AutoClustering:
    def __init__(self, n_clusters, numeric_features=None, categorical_features=None, random_state=42):
        self.n_clusters = n_clusters
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.random_state = random_state
        self.model = None
        self.method_used = None

    def fit(self, df):
        # Détection automatique du type
        if len(self.categorical_features)==0:
            # tout numérique → KMeans
            self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            self.method_used = "KMeans"
            X = df[self.numeric_features].values
            clusters = self.model.fit_predict(X)
            prototypes = self.model.cluster_centers_
        elif len(self.numeric_features)==0:
            # tout catégoriel → KModes
            self.model = KModes(n_clusters=self.n_clusters, random_state=self.random_state, init='Huang')
            self.method_used = "KModes"
            X = df[self.categorical_features].values
            clusters = self.model.fit_predict(X)
            prototypes = np.array(self.model.cluster_centroids_, dtype=object)
        else:
            # mixte → KPrototypes
            self.model = KPrototypes(n_clusters=self.n_clusters, random_state=self.random_state, init='Huang')
            self.method_used = "KPrototypes"
            X = df[self.numeric_features + self.categorical_features].values
            cat_idx = [df.columns.get_loc(c) for c in self.categorical_features]
            clusters = self.model.fit_predict(X, categorical=cat_idx)
            prototypes = np.array(self.model.cluster_centroids_, dtype=object)

        result_df = df.copy()
        result_df['cluster'] = clusters

        model_info = {
            'method': self.method_used,
            'prototypes': prototypes,
            'prototypes_cols': self.numeric_features + self.categorical_features
        }

        return result_df, model_info

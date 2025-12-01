# clustering.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Algorithms
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# kmodes package: KModes and KPrototypes
# ensure kmodes is in requirements
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

class AutoClustering:
    def __init__(self, n_clusters:int,
                 numeric_features:List[str],
                 categorical_features:List[str],
                 random_state:int=42):
        self.n_clusters = n_clusters
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.random_state = random_state

        self.model = None
        self.method = None

    def _all_numeric(self):
        return len(self.categorical_features) == 0 and len(self.numeric_features) > 0

    def _all_categorical(self):
        return len(self.numeric_features) == 0 and len(self.categorical_features) > 0

    def fit(self, df:pd.DataFrame):
        """
        Detects type, runs appropriate algorithm and returns
        - df with 'cluster' column
        - info dict {method, prototypes (DataFrame), model (object), extra}
        """
        df_copy = df.copy().reset_index(drop=True)

        # Case 1: all numeric -> KMeans
        if self._all_numeric():
            X = df_copy[self.numeric_features].astype(float).values
            km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            labels = km.fit_predict(X)
            df_copy['cluster'] = labels
            # prototypes = centroids
            proto_df = pd.DataFrame(km.cluster_centers_, columns=self.numeric_features)
            info = {'method':'kmeans', 'model':km, 'prototypes': proto_df}
            return df_copy, info

        # Case 2: all categorical -> KModes
        if self._all_categorical():
            # KModes expects numpy array of categorical values
            X = df_copy[self.categorical_features].astype(str).values
            kmodes = KModes(n_clusters=self.n_clusters, init='Cao', n_init=5, random_state=self.random_state)
            labels = kmodes.fit_predict(X)
            df_copy['cluster'] = labels
            # prototypes: modes per cluster
            modes = []
            for k in range(self.n_clusters):
                cluster_rows = X[labels == k]
                if len(cluster_rows) == 0:
                    modes.append([None]*len(self.categorical_features))
                else:
                    # column-wise mode
                    col_modes = []
                    for col_i in range(cluster_rows.shape[1]):
                        vals, counts = np.unique(cluster_rows[:, col_i], return_counts=True)
                        col_modes.append(vals[np.argmax(counts)])
                    modes.append(col_modes)
            proto_df = pd.DataFrame(modes, columns=self.categorical_features)
            info = {'method':'kmodes', 'model':kmodes, 'prototypes': proto_df}
            return df_copy, info

        # Case 3: mixed -> KPrototypes
        # Build numpy array where categorical columns are strings
        mixed_cols = self.numeric_features + self.categorical_features
        X_mixed = df_copy[mixed_cols].copy()
        # KPrototypes wants numpy array with categorical columns as strings
        for c in self.categorical_features:
            X_mixed[c] = X_mixed[c].astype(str)
        # Convert to numpy
        X_np = X_mixed.values
        # categorical indices relative to X_np
        cat_indices = [mixed_cols.index(c) for c in self.categorical_features]

        kproto = KPrototypes(n_clusters=self.n_clusters, init='Cao', verbose=0, random_state=self.random_state)
        labels = kproto.fit_predict(X_np, categorical=cat_indices)
        df_copy['cluster'] = labels

        # Build prototypes DataFrame (centroid numeric + mode categorical)
        protos = []
        for r in kproto.cluster_centroids_:
            # r is a mixed-type array (strings for categorical)
            protos.append(list(r))
        proto_df = pd.DataFrame(protos, columns=mixed_cols)

        info = {'method':'kprototypes', 'model':kproto, 'prototypes': proto_df, 'cat_indices':cat_indices}
        return df_copy, info

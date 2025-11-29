import numpy as np
import pandas as pd

class DynamicClusteringGeneralized:
    def __init__(self, num_clusters, numeric_features, categorical_features,
                 max_iterations=100, tolerance=1e-4):

        self.num_clusters = num_clusters
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.col_order = None
        self.numeric_idx = None
        self.categorical_idx = None
        
        self.cluster_prototypes = None

    # -------------------------
    #  PREPARE DATA
    # -------------------------
    def _prepare_data(self, data):

        df = pd.DataFrame(data)
        self.col_order = list(df.columns)

        self.numeric_idx = [self.col_order.index(col) for col in self.numeric_features]
        self.categorical_idx = [self.col_order.index(col) for col in self.categorical_features]

        return df

    # -------------------------
    #  GOWER DISTANCE
    # -------------------------
    def _gower_distance(self, x, y):

        dist = 0
        count = 0

        # Numerical part
        for idx in self.numeric_idx:
            rng = self.numeric_ranges[idx]
            dist += abs(x[idx] - y[idx]) / (rng + 1e-9)
            count += 1

        # Categorical part
        for idx in self.categorical_idx:
            dist += 0 if x[idx] == y[idx] else 1
            count += 1

        return dist / count

    # -------------------------
    #  PROTOTYPES IN ORIGINAL COLUMN ORDER
    # -------------------------
    def _compute_prototypes(self, df, assignments):

        prototypes = []

        for k in range(self.num_clusters):
            cluster_points = df[assignments == k]

            if cluster_points.empty:
                # sample one row and convert to native Python types
                sampled = df.sample(1).iloc[0]
                proto = []
                for col in self.col_order:
                    if col in self.numeric_features:
                        val = sampled[col]
                        if pd.isna(val):
                            proto.append(None)
                        else:
                            f = float(val)
                            proto.append(int(f) if f.is_integer() else f)
                    elif col in self.categorical_features:
                        proto.append(sampled[col])
                prototypes.append(np.array(proto, dtype=object))
                continue

            proto = []

            for col in self.col_order:
                if col in self.numeric_features:
                    val = cluster_points[col].mean()
                    # convert numpy numeric to native Python int/float
                    if pd.isna(val):
                        proto.append(None)
                    else:
                        f = float(val)
                        proto.append(int(f) if f.is_integer() else f)
                elif col in self.categorical_features:
                    mode_val = cluster_points[col].mode()[0]
                    proto.append(mode_val)

            prototypes.append(np.array(proto, dtype=object))

        return np.array(prototypes)

    # -------------------------
    #  ASSIGN POINTS
    # -------------------------
    def _assign_clusters(self, df, prototypes):

        assignments = []

        for row in df.values:
            dists = [self._gower_distance(row, proto) for proto in prototypes]
            assignments.append(np.argmin(dists))

        return np.array(assignments)

    # -------------------------
    #  FIT
    # -------------------------
    def fit(self, data):

        df = self._prepare_data(data)

        # Compute numeric ranges
        self.numeric_ranges = {
            self.col_order.index(col): df[col].max() - df[col].min()
            for col in self.numeric_features
        }

        assignments = np.random.randint(0, self.num_clusters, size=len(df))

        for it in range(self.max_iterations):

            new_prototypes = self._compute_prototypes(df, assignments)
            new_assignments = self._assign_clusters(df, new_prototypes)

            if np.array_equal(assignments, new_assignments):
                print(f"Convergence reached at iteration {it}")
                break

            assignments = new_assignments
            self.cluster_prototypes = new_prototypes

        return assignments

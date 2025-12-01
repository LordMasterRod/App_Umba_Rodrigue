import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def run_clustering(df, features, k):
    X = df[features]

    model = KMeans(n_clusters=k, random_state=42)
    df["cluster"] = model.fit_predict(X)

    return df, model


def plot_clusters(df, features, model):
    x_col, y_col = features[0], features[1]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Points du dataset
    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        c=df["cluster"],
        s=50
    )

    # Centroïdes
    centroids = model.cluster_centers_
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=200,
        marker="X",
        edgecolor="black",
        linewidth=2
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Nuée dynamique – K-Means")

    return fig

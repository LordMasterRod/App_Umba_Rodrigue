# plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_2d_scatter(df, x_col, y_col, cluster_col='cluster', prototypes=None, prototype_cols=None):
    """
    Plot 2D scatter with clusters and optional prototypes.
    df : DataFrame with points and cluster assignment
    x_col, y_col : numeric columns for axes
    cluster_col : name of column with cluster assignment
    prototypes : np.array or DataFrame with cluster centers
    prototype_cols : list with [x_col, y_col] for prototypes
    """
    fig, ax = plt.subplots(figsize=(8,6))
    palette = sns.color_palette("tab10", n_colors=df[cluster_col].nunique())

    # Plot points
    for cl in sorted(df[cluster_col].unique()):
        pts = df[df[cluster_col] == cl]
        ax.scatter(pts[x_col], pts[y_col], s=50, alpha=0.7, label=f'Cluster {cl}', color=palette[cl])

    # Plot prototypes if provided
    if prototypes is not None and prototype_cols is not None:
        ax.scatter(
            prototypes[:, 0], prototypes[:, 1],
            c='red', marker='X', s=200, label='Prototypes'
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    ax.grid(True)
    return fig

def plot_categorical_summary(df, cat_col):
    """
    Bar plot showing distribution of categorical column per cluster.
    """
    cluster_col = 'cluster'
    summary = df.groupby([cluster_col, cat_col]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8,6))
    summary.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {cat_col} per cluster')
    ax.legend(title=cat_col)
    return fig

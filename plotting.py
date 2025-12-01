# plotting.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def plot_2d_scatter(df:pd.DataFrame, x_col:str, y_col:str, info:dict):
    fig, ax = plt.subplots(figsize=(8,6))
    clusters = df['cluster'].unique()
    cmap = plt.get_cmap('tab10')

    for i, cl in enumerate(sorted(clusters)):
        pts = df[df['cluster']==cl]
        ax.scatter(pts[x_col], pts[y_col], label=f'Cluster {cl}', alpha=0.7, s=50, cmap=cmap)

    # prototypes exist in info['prototypes']
    proto = info.get('prototypes')
    if proto is not None and x_col in proto.columns and y_col in proto.columns:
        ax.scatter(proto[x_col], proto[y_col], color='red', marker='X', s=220, edgecolors='black', linewidth=1.2, label='Prototypes')

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title('Clusters - 2D scatter')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_categorical_summary(df:pd.DataFrame, cat_col:str):
    """
    For categorical-only case: plot bar counts per cluster for this categorical column
    """
    fig, ax = plt.subplots(figsize=(8,4))
    ct = pd.crosstab(df['cluster'], df[cat_col])
    ct.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'Distribution of {cat_col} by cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')
    plt.tight_layout()
    return fig

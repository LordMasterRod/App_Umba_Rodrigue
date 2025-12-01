# plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_2d_scatter(df, x_col, y_col, model_info, show_points=True, show_prototypes=True):
    fig, ax = plt.subplots(figsize=(8,6))
    
    # --- Scatter points ---
    if show_points:
        clusters = sorted(df['cluster'].unique())
        colors = sns.color_palette('tab10', n_colors=len(clusters))
        for cl, color in zip(clusters, colors):
            pts = df[df['cluster']==cl]
            if pts.empty:
                continue
            ax.scatter(
                pts[x_col],
                pts[y_col],
                label=f'Cluster {cl}',
                alpha=0.7,
                s=50,
                color=color
            )
    
    # --- Scatter prototypes ---
    if show_prototypes:
        prototypes = model_info['prototypes']
        proto_cols = model_info['prototypes_cols']
        x_idx = proto_cols.index(x_col)
        y_idx = proto_cols.index(y_col)
        
        ax.scatter(
            prototypes[:, x_idx],
            prototypes[:, y_idx],
            color='red',
            s=200,
            marker='X',
            label='Prototypes'
        )
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Clusters + Prototypes")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_categorical_summary(df, cat_col):
    fig, ax = plt.subplots(figsize=(8,5))
    pd.crosstab(df['cluster'], df[cat_col]).plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution de {cat_col} par cluster")
    plt.tight_layout()
    return fig

# App.py
import streamlit as st
import pandas as pd
import numpy as np

from clustering import AutoClustering
from plotting import plot_2d_scatter, plot_categorical_summary

st.set_page_config(page_title="Auto Clustering (KMeans / KModes / KPrototypes)", layout="wide")
st.title("🟦 Auto Clustering — KMeans / KModes / KPrototypes (Automatic)")

st.write("""
L'application détecte automatiquement le **type** des colonnes (numériques / catégorielles)
et choisit l'algorithme adapté :
- **KMeans** pour tout numérique,
- **KModes** pour tout catégoriel,
- **KPrototypes** pour données mixtes.
""")

# --- Sidebar: upload dataset ---
st.sidebar.header("📁 Charger un dataset (CSV)")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# --- Load sample if no upload ---
if not uploaded:
    st.info("Dépose un fichier CSV pour commencer. Exemple inclus : `dataset_mixed.csv`.")
    sample = pd.read_csv("dataset_mixed.csv")
    st.subheader("Dataset d'exemple (aperçu)")
    st.dataframe(sample.head())
    st.stop()

# --- Load uploaded data ---
df = pd.read_csv(uploaded)
st.subheader("📄 Aperçu des données")
st.dataframe(df.head())

# --- Auto detect feature types ---
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

st.sidebar.markdown("### 🔎 Détection automatique des types")
st.sidebar.write(f"Numeric cols detected: {numeric_cols}")
st.sidebar.write(f"Categorical cols detected: {categorical_cols}")

# --- Allow user override ---
st.sidebar.markdown("### ✏️ Override (optionnel)")
user_numeric = st.sidebar.multiselect("Sélectionner colonnes numériques", options=df.columns.tolist(), default=numeric_cols)
user_categorical = st.sidebar.multiselect("Sélectionner colonnes catégorielles", options=df.columns.tolist(), default=categorical_cols)

use_numeric = user_numeric if user_numeric else numeric_cols
use_categorical = user_categorical if user_categorical else categorical_cols

st.sidebar.markdown("---")
n_clusters = st.sidebar.slider("Nombre de clusters (k)", 2, 10, 3)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42)

# --- Run clustering ---
if st.button("🚀 Run clustering"):
    with st.spinner("Running clustering..."):
        auto = AutoClustering(
            n_clusters=n_clusters,
            numeric_features=use_numeric,
            categorical_features=use_categorical,
            random_state=int(random_state)
        )

        result_df, model_info = auto.fit(df)

    st.success(f"Clustering done — method: {model_info['method']}")

    # --- Show assignments ---
    st.subheader("🔖 Assignations (index -> cluster)")
    st.dataframe(result_df[['cluster']].reset_index())

    # --- Show prototypes/centroids/modes ---
    st.subheader("📌 Prototypes / Centroids / Modes")
    st.dataframe(model_info['prototypes'])

    st.divider()

    # --- Visualization ---
    st.subheader("📈 Visualisation des clusters")

    # If at least 2 numeric features available for plotting
    if len(use_numeric) >= 2:
        x_col = st.selectbox("X axis", use_numeric, index=0)
        y_col = st.selectbox("Y axis", use_numeric, index=1 if len(use_numeric) > 1 else 0)

        fig = plot_2d_scatter(
            result_df,
            x_col,
            y_col,
            model_info,
            show_points=True,        # <-- affichage points
            show_prototypes=True     # <-- affichage prototypes
        )
        st.pyplot(fig)

    else:
        # For categorical only: show distributions per cluster
        st.info("Moins de 2 colonnes numériques : affichage des distributions catégorielles par cluster.")
        for cat in use_categorical:
            fig = plot_categorical_summary(result_df, cat)
            st.pyplot(fig)

    # --- Download results ---
    st.download_button(
        "⬇️ Télécharger résultats (CSV)",
        data=result_df.to_csv(index=False),
        file_name="clustering_results.csv",
        mime="text/csv"
    )

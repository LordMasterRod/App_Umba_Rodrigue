import streamlit as st
import pandas as pd
from Clustering import run_clustering, plot_clusters

st.set_page_config(page_title="Clustering App", layout="wide")

st.title("🔵 Machine Learning – Clustering (K-Means)")

st.write(
    "Cette application permet d’effectuer un clustering K-Means, "
    "d’afficher les clusters et la nuée dynamique avec les centroïdes."
)

# ==== UPLOAD DATA ====
st.sidebar.header("📁 Chargement des données")
uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📌 Aperçu du dataset")
    st.dataframe(df.head())

    # Selection variables
    st.sidebar.header("⚙ Paramètres du modèle")
    columns = df.columns.tolist()
    selected_features = st.sidebar.multiselect(
        "Sélectionner les colonnes à utiliser", columns, default=columns[:2]
    )

    k = st.sidebar.number_input(
        "Nombre de clusters (k)", min_value=2, max_value=10, value=3
    )

    if st.sidebar.button("Lancer le clustering"):
        if len(selected_features) < 2:
            st.error("Sélectionne au moins 2 colonnes.")
        else:
            st.success("Clustering effectué avec succès !")

            df_result, model = run_clustering(df, selected_features, k)

            st.subheader("📊 Résultats du clustering")
            st.dataframe(df_result.head())

            # Plot
            st.subheader("📌 Nuée dynamique (clusters + centroïdes)")
            fig = plot_clusters(df_result, selected_features, model)
            st.pyplot(fig)

else:
    st.info(
        "Veuillez importer un fichier CSV depuis le menu latéral. "
        "Un dataset d’exemple est proposé ci-dessous."
    )

    sample = pd.read_csv("dataset.csv")
    st.dataframe(sample.head())

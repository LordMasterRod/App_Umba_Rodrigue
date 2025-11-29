# app.py
import streamlit as st
import pandas as pd
from clustering import DynamicClusteringGeneralized

st.title("Nuée Dynamique - Clustering de Données Mixtes")

# Chargement dataset
st.sidebar.header("Paramètres du dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu du dataset :")
    st.dataframe(data.head())
    
    # Définir colonnes numériques et catégorielles
    numeric_cols = st.multiselect("Colonnes numériques", options=data.columns.tolist())
    categorical_cols = st.multiselect("Colonnes catégorielles", options=data.columns.tolist())
    
    num_clusters = st.slider("Nombre de clusters", min_value=2, max_value=10, value=2)

    if st.button("Lancer clustering"):

        model = DynamicClusteringGeneralized(
            num_clusters=num_clusters,
            numeric_features=numeric_cols,
            categorical_features=categorical_cols
        )

        assignments = model.fit(data)
        st.subheader("Résultats :")
        st.write("Cluster assignments :")
        st.write(assignments)
        
        st.write("Prototypes de clusters :")
        st.write(pd.DataFrame(model.cluster_prototypes, columns=numeric_cols + categorical_cols))
else:
    st.info("Upload ton fichier CSV pour commencer.")

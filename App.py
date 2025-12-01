import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from Clustering import DynamicClusteringGeneralized

# ================================
#   CONFIGURATION DE LA PAGE
# ================================
st.set_page_config(
    page_title="Nuée Dynamique - Mixed Clustering",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Nuée Dynamique – Clustering sur Données Mixtes")

st.write("""
Cette application permet d'exécuter **l'algorithme de nuée dynamique** (version généralisée)  
sur un dataset contenant des **variables numériques et catégorielles**.
""")

st.divider()

# ================================
#   UPLOAD DU CSV
# ================================
st.sidebar.header("📁 Charger un Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if not uploaded_file:
    st.info("➡️ Charge un fichier CSV pour commencer.")
    st.stop()

# Charger dataset
data = pd.read_csv(uploaded_file)
st.subheader("📄 Aperçu du dataset")
st.dataframe(data.head())

st.divider()

# ================================
#   SÉLECTION DES COLONNES
# ================================
st.sidebar.header("⚙️ Paramètres du Clustering")

numeric_cols = st.sidebar.multiselect(
    "Colonnes numériques",
    options=data.columns.tolist(),
    default=[col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
)

categorical_cols = st.sidebar.multiselect(
    "Colonnes catégorielles",
    options=data.columns.tolist(),
    default=[col for col in data.columns if data[col].dtype == "object"]
)

num_clusters = st.sidebar.slider("Nombre de clusters", 2, 10, 3)

# Sécurité
if len(numeric_cols) == 0 and len(categorical_cols) == 0:
    st.error("❌ Vous devez sélectionner au moins une colonne numérique ou catégorielle.")
    st.stop()

st.divider()

# ================================
#   BOUTON : LANCER LE CLUSTERING
# ================================
if st.button("🚀 Lancer la Nuée Dynamique"):

    with st.spinner("Clustering en cours..."):

        model = DynamicClusteringGeneralized(
            num_clusters=num_clusters,
            numeric_features=numeric_cols,
            categorical_features=categorical_cols
        )

        assignments = model.fit(data)

    st.success("Clustering terminé !")

    # ================================
    #   RÉSULTATS DU CLUSTERING
    # ================================
    st.subheader("📌 Assignations des Clusters")
    st.dataframe(pd.DataFrame({
        "Index": data.index,
        "Cluster": assignments
    }))

    # Prototypes
    st.subheader("📌 Prototypes des Clusters (Étalons)")
    try:
        proto_df = pd.DataFrame(model.cluster_prototypes, columns=numeric_cols + categorical_cols)
        st.dataframe(proto_df)
    except:
        st.warning("Les prototypes ne peuvent pas être affichés correctement.")

    st.divider()

    # ================================
    #   SCATTER PLOT 2D
    # ================================
    st.subheader("📈 Visualisation 2D des Clusters")

    if len(numeric_cols) < 2:
        st.warning("⚠️ Il faut au moins **2 colonnes numériques** pour afficher un graphique.")
        st.stop()

    # Choix dynamique des axes
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Axe X :", numeric_cols, index=0)
    with col2:
        y_axis = st.selectbox("Axe Y :", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster_id in range(num_clusters):
        pts = data[assignments == cluster_id]
        ax.scatter(
            pts[x_axis], pts[y_axis],
            alpha=0.7,
            label=f"Cluster {cluster_id}"
        )

    # Prototypes (rouge)
    proto_df = pd.DataFrame(model.cluster_prototypes, columns=numeric_cols + categorical_cols)
    ax.scatter(
        proto_df[x_axis],
        proto_df[y_axis],
        color="red",
        marker="X",
        s=250,
        label="Prototypes",
        edgecolors="black"
    )

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title("Nuée Dynamique – Projection 2D")
    ax.legend()

    st.pyplot(fig)

else:
    st.info("Clique sur **Lancer la Nuée Dynamique** pour exécuter l’algorithme.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# =====================================
# TITLE & DESCRIPTION
# =====================================
st.title("🟢 Customer Segmentation Dashboard")
st.markdown(
    """
    **This system uses K-Means Clustering to group customers based on their purchasing behavior and similarities.**

    """
)



df = pd.read_csv('Wholesale customers data.csv')

# =====================================
# SIDEBAR: CLUSTERING INPUT CONTROLS
# =====================================
st.sidebar.header("⚙️ Clustering Controls")

# Select only numerical columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Dataset must contain at least two numerical columns.")
    st.stop()

feature_1 = st.sidebar.selectbox("Select Feature 1", numeric_cols)
feature_2 = st.sidebar.selectbox(
    "Select Feature 2",
    [col for col in numeric_cols if col != feature_1]
)

k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)

random_state = st.sidebar.number_input(
    "Random State (Optional)", value=42, step=1
)

run_button = st.sidebar.button("🟦 Run Clustering")

# =====================================
# RUN CLUSTERING
# =====================================
if run_button:

    # Prepare data
    X = df[[feature_1, feature_2]].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means model
    kmeans = KMeans(n_clusters=k, random_state=int(random_state))
    clusters = kmeans.fit_predict(X_scaled)

    df["Cluster"] = clusters

    # =====================================
    # VISUALIZATION SECTION
    # =====================================
    st.subheader("📊 Customer Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=clusters,
        cmap="viridis",
        alpha=0.7
    )

    centers = kmeans.cluster_centers_
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="red",
        s=200,
        marker="X",
        label="Cluster Centers"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("K-Means Customer Segmentation")
    ax.legend()

    st.pyplot(fig)

    # =====================================
    # CLUSTER SUMMARY
    # =====================================
    st.subheader("📋 Cluster Summary")

    cluster_summary = (
        df.groupby("Cluster")
        .agg(
            Count=("Cluster", "count"),
            Avg_Feature_1=(feature_1, "mean"),
            Avg_Feature_2=(feature_2, "mean")
        )
        .reset_index()
    )

    st.dataframe(cluster_summary)

    # =====================================
    # BUSINESS INTERPRETATION
    # =====================================
    st.subheader("💼 Business Interpretation")

    for _, row in cluster_summary.iterrows():
        cluster_id = int(row["Cluster"])
        count = row["Count"]

        avg1 = row["Avg_Feature_1"]
        avg2 = row["Avg_Feature_2"]

        if avg1 > X[feature_1].mean() and avg2 > X[feature_2].mean():
            insight = "High-spending customers across selected categories."
            icon = "🟢"
        elif avg1 < X[feature_1].mean() and avg2 < X[feature_2].mean():
            insight = "Budget-conscious customers with lower spending."
            icon = "🟡"
        else:
            insight = "Moderate spenders with selective purchasing behavior."
            icon = "🔵"

        st.markdown(
            f"{icon} **Cluster {cluster_id}** ({count} customers): {insight}"
        )

    # =====================================
    # USER GUIDANCE
    # =====================================
    st.info(
        "Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
    )

else:
    st.info("👈 Select features and click **Run Clustering** to generate results.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(layout="wide")
st.title("🧠 Clustering Dashboard")

st.markdown("""
Welcome to the **Clustering Dashboard**! This app allows you to upload your dataset, apply different clustering algorithms, and visualize the results interactively.

### Key Features:
- **Scaling** of features using StandardScaler.
- **PCA** for dimensionality reduction (2D visualization).
- **Interactive Clustering** with K-Means, DBSCAN, and Hierarchical Clustering.
- **Feature Contributions** displayed in PCA components to interpret the results.
- **New User Cluster Prediction**
""")

uploaded_file = st.file_uploader("📅 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Raw Data Preview")
    st.dataframe(df.head())
    st.write("### Summary Statistics")
    st.write(df.describe())

    original_categorical_cols = df.select_dtypes(include='object').columns.tolist()
    label_encoders = {}

    for col in original_categorical_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        label_encoders[col] = le

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect("🔧 Select features for clustering:", numeric_cols, default=numeric_cols[:2])

    if len(selected_features) >= 2:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[selected_features])

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        df["PCA1"] = pca_data[:, 0]
        df["PCA2"] = pca_data[:, 1]

        st.subheader("🧬 PCA Feature Contributions")
        pca_components_df = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=selected_features)
        st.dataframe(pca_components_df.style.format("{:.2f}"))

        fig_pca, ax_pca = plt.subplots(figsize=(10, 5))
        pca_components_df.plot(kind="bar", ax=ax_pca)
        ax_pca.set_title("Feature Contributions to Principal Components")
        ax_pca.set_ylabel("Loading Weight")
        ax_pca.set_xlabel("Features")
        ax_pca.axhline(0, color='gray', linewidth=0.8)
        plt.xticks(rotation=45)
        st.pyplot(fig_pca)

        use_pca = st.checkbox("🧪 Use PCA-transformed data for clustering and visualization", value=True)
        data_for_clustering = pca_data if use_pca else scaled_data

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 K-Means Clustering")
            k = st.slider("Number of Clusters (KMeans)", 2, 10, 3)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df["KMeans_Cluster"] = kmeans.fit_predict(data_for_clustering)

            st.write("Cluster counts (KMeans):")
            st.dataframe(df["KMeans_Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

            fig1, ax1 = plt.subplots()
            sns.scatterplot(
                x=df["PCA1"] if use_pca else df[selected_features[0]],
                y=df["PCA2"] if use_pca else df[selected_features[1]],
                hue=df["KMeans_Cluster"],
                palette="viridis",
                ax=ax1
            )
            ax1.set_title("K-Means Clustering")
            st.pyplot(fig1)

        with col2:
            st.subheader("🌌 DBSCAN Clustering")
            eps = st.slider("EPS (Neighborhood radius)", 0.1, 100.0, 10.0)
            min_samples = st.slider("Min Samples", 1, 10, 2)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            df["DBSCAN_Cluster"] = dbscan.fit_predict(data_for_clustering)

            st.write("Cluster counts (DBSCAN):")
            st.dataframe(df["DBSCAN_Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

            fig2, ax2 = plt.subplots()
            sns.scatterplot(
                x=df["PCA1"] if use_pca else df[selected_features[0]],
                y=df["PCA2"] if use_pca else df[selected_features[1]],
                hue=df["DBSCAN_Cluster"],
                palette="plasma",
                ax=ax2
            )
            ax2.set_title("DBSCAN Clustering")
            st.pyplot(fig2)

        st.subheader("🧬 Hierarchical Clustering (Dendrogram)")
        linkage_matrix = linkage(data_for_clustering, method="ward")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        dendrogram(linkage_matrix, ax=ax3)
        ax3.set_title("Dendrogram (Ward linkage)")
        ax3.set_xlabel("Data Points")
        ax3.set_ylabel("Distance")
        st.pyplot(fig3)

        n_hier_clusters = st.slider("Number of Clusters (Agglomerative)", 2, 10, 3)
        hier = AgglomerativeClustering(n_clusters=n_hier_clusters)
        df["Hierarchical_Cluster"] = hier.fit_predict(data_for_clustering)

        st.write("Cluster counts (Hierarchical):")
        st.dataframe(df["Hierarchical_Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

        st.subheader("📊 Clustered Data Preview")
        st.dataframe(df.head())

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Clustered Data as CSV",
            data=csv,
            file_name='clustered_data.csv',
            mime='text/csv'
        )

        # Predict cluster for a new user
        st.subheader("🔍 Predict Cluster for a New User")

        new_user_inputs = {}
        with st.form("new_user_form"):
            st.markdown("#### Enter values for the new user")

            # Categorical features
            for col in original_categorical_cols:
                unique_vals = df[col].unique().tolist()
                new_user_inputs[col] = st.selectbox(f"{col}", unique_vals)

            # Numeric-only selected features
            for col in selected_features:
                if col not in df.columns or col.endswith("_enc"):
                    continue
                new_user_inputs[col] = st.number_input(f"{col}", value=float(df[col].mean()))

            submitted = st.form_submit_button("Predict Cluster")

            if submitted:
                new_user_df = pd.DataFrame([new_user_inputs])

                # Encode categorical features
                for col in original_categorical_cols:
                    le = label_encoders[col]
                    new_user_df[col + "_enc"] = le.transform(new_user_df[col])

                # Build feature vector for clustering
                input_features = []
                for col in selected_features:
                    if col in new_user_df.columns:
                        input_features.append(new_user_df[col])
                    elif col.endswith("_enc") and col in new_user_df.columns:
                        input_features.append(new_user_df[col])
                    elif col in df.columns:
                        input_features.append(new_user_df[col])

                input_data = pd.concat(input_features, axis=1)
                input_data.columns = selected_features

                scaled_input = scaler.transform(input_data)
                input_pca = pca.transform(scaled_input) if use_pca else scaled_input
                predicted_cluster = kmeans.predict(input_pca)[0]

                st.success(f"✅ The new user belongs to KMeans Cluster: {predicted_cluster}")

    else:
        st.warning("Please select at least two numeric features for clustering.")
else:
    st.info("📂 Upload a CSV file to get started.")

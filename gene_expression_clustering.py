import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Simulate a dataset with multiple gene entries
np.random.seed(42)
num_samples = 150

data = {
    "geneId": np.random.randint(100000, 999999, size=num_samples),
    "proteinCount": np.random.randint(0, 10, size=num_samples),
    "transcriptCount": np.random.randint(1, 6, size=num_samples),
    "taxId": np.random.choice([9606, 10090, 10116], size=num_samples),  # Human, Mouse, Rat
}

# Wrap in a DataFrame
df = pd.DataFrame(data)

# Step 1: Basic Inspection
print("Synthetic DataFrame:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Step 2: Drop missing values (just for consistency)
df_clean = df.dropna()

# Step 3: Select only numeric columns
df_numeric = df_clean.select_dtypes(include='number')
print(f"\nShape after cleaning: {df_numeric.shape}")

# Step 4: PCA to reduce to 2 dimensions for visualization
if df_numeric.shape[0] > 1:
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_numeric)
    df_pca = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])

    # Step 5: K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_pca['KMeans_Cluster'] = kmeans.fit_predict(reduced_data)
    print("\nKMeans Cluster Counts:")
    print(df_pca['KMeans_Cluster'].value_counts())

    # Step 6: DBSCAN Clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(reduced_data)

    dbscan = DBSCAN(eps=0.5, min_samples=2)
    df_pca['DBSCAN_Cluster'] = dbscan.fit_predict(scaled_data)
    print("\nDBSCAN Cluster Counts:")
    print(df_pca['DBSCAN_Cluster'].value_counts())

    # Step 7: Plot KMeans Clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='KMeans_Cluster', data=df_pca, palette='Set1')
    plt.title("KMeans Clustering of Synthetic Genes")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig("kmeans_clusters.png")
    plt.show()

    # Step 8: Plot DBSCAN Clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='DBSCAN_Cluster', data=df_pca, palette='Set2')
    plt.title("DBSCAN Clustering of Synthetic Genes")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig("dbscan_clusters.png")
    plt.show()
else:
    print("❌ Not enough samples for PCA and clustering. Need at least 2 rows of data.")


# If you later want to switch back to using data_report.jsonl with real data:

# Replace the synthetic data block with:

# python
# Copy code
# df = pd.read_json("data_report.jsonl", lines=True)
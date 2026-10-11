
# Day 13: Hierarchical Clustering

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
data = load_iris()
X = data.data[:, [0, 1]]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create hierarchical clustering model
model = AgglomerativeClustering(n_clusters=3, linkage="ward")
clusters = model.fit_predict(X_scaled)

# Print results
print("Hierarchical Clustering Results")
print("Total samples:", len(X_scaled))
print("Number of clusters:", len(set(clusters)))

print("\nSamples in each cluster:")
for cluster_id in range(3):
    count = (clusters == cluster_id).sum()
    print(f"Cluster {cluster_id}: {count} samples")

# Plot clusters
plt.figure(figsize=(8, 5))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters,
    cmap="viridis",
    s=50
)
plt.title("Hierarchical Clustering on Iris Dataset")
plt.xlabel("Standardized Sepal Length")
plt.ylabel("Standardized Sepal Width")
plt.tight_layout()
plt.show()

# Plot dendrogram using a sample of 30 observations
Z = linkage(X_scaled[:30], method="ward")

plt.figure(figsize=(12, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram (First 30 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

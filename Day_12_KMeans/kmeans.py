
# Day 12: K-Means Clustering

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data[:, [0, 1]]  # Sepal length and sepal width

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model
model = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train the model without using target labels
clusters = model.fit_predict(X_scaled)

# Print results
print("K-Means Clustering Results")
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

plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("K-Means Clustering on Iris Dataset")
plt.xlabel("Standardized Sepal Length")
plt.ylabel("Standardized Sepal Width")
plt.legend()
plt.tight_layout()
plt.show()

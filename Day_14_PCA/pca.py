
# Day 14: Principal Component Analysis (PCA)

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Original dataset shape:", X.shape)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce 30 features to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Reduced dataset shape:", X_pca.shape)

# Print explained variance
print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

total_variance = sum(pca.explained_variance_ratio_)
print("Total variance retained:", round(total_variance * 100, 2), "%")

# Visualize the principal components
plt.figure(figsize=(8, 5))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    s=40
)

plt.title("PCA - Breast Cancer Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=list(data.target_names),
    title="Actual class"
)

plt.tight_layout()
plt.show()


# Day 12 - K-Means Clustering

## Objective
Learn unsupervised machine learning using K-Means clustering.

## Dataset
Iris dataset from Scikit-learn.

- Total samples: 150
- Features used: Sepal length and sepal width
- Number of clusters: 3

## Concepts Learned
- Unsupervised learning
- Clustering
- Centroids
- Feature scaling
- Cluster visualization

## Algorithm
K-Means groups data points into K clusters by assigning points to the nearest centroid and updating the centroids iteratively.

## Steps Performed
1. Loaded the Iris dataset.
2. Selected two numerical features.
3. Scaled the features using StandardScaler.
4. Created a K-Means model with three clusters.
5. Assigned each sample to a cluster.
6. Visualized the clusters and centroids.

## Visualization
The scatter plot displays the three clusters and their centroids.

## Libraries Used
- Scikit-learn
- Matplotlib

## How to Run
python kmeans.py

## Conclusion
This practical introduced unsupervised learning and showed how K-Means groups similar data points without using class labels during training.

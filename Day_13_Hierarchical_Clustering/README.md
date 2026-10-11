
# Day 13 - Hierarchical Clustering

## Objective
Learn unsupervised machine learning using hierarchical clustering.

## Dataset
Iris dataset from Scikit-learn.

- Total samples: 150
- Features used: Sepal length and sepal width
- Number of clusters: 3

## Concepts Learned
- Unsupervised learning
- Agglomerative clustering
- Hierarchical clustering
- Dendrograms
- Feature scaling
- Cluster visualization

## Algorithm
Agglomerative clustering starts with individual data points and repeatedly merges the closest clusters until the desired number of clusters remains.

## Steps Performed
1. Loaded the Iris dataset.
2. Selected two numerical features.
3. Scaled the features using StandardScaler.
4. Applied Agglomerative Clustering.
5. Visualized the clusters.
6. Created a dendrogram for the first 30 observations.

## Libraries Used
- Scikit-learn
- Matplotlib
- SciPy

## How to Run
python hierarchical_clustering.py

## Conclusion
This practical introduced hierarchical clustering and dendrograms to understand how data points can be grouped based on similarity.

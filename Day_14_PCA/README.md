
# Day 14 - Principal Component Analysis (PCA)

## Objective
Learn dimensionality reduction using PCA.

## Dataset
Breast Cancer Wisconsin dataset from Scikit-learn.

- Total samples: 569
- Original features: 30
- Reduced components: 2

## Concepts Learned
- Dimensionality reduction
- Feature scaling
- Principal components
- Explained variance
- Data visualization

## Algorithm
PCA transforms the original features into new components that capture as much of the data's variance as possible. The first component captures the greatest variance, followed by the next component.

## Steps Performed
1. Loaded the dataset.
2. Standardized the features.
3. Reduced 30 features to 2 principal components.
4. Printed the explained variance ratio.
5. Visualized the data in a 2D scatter plot.

## Results
Record the explained variance and total variance retained from the program output.

## Libraries Used
- Scikit-learn
- Matplotlib

## How to Run
python pca.py

## Conclusion
This practical demonstrated how PCA reduces the number of features and helps visualize high-dimensional data in two dimensions.

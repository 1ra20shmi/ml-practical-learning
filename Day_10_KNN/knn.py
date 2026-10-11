
# Day 10: K-Nearest Neighbors (KNN)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create KNN model
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("KNN Results")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# Graph 1: Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=data.target_names,
    cmap="Blues"
)

plt.title("KNN - Confusion Matrix")
plt.tight_layout()
plt.show()


# Graph 2: Accuracy Comparison
models = [
    "Logistic Regression",
    "SVM",
    "KNN"
]

accuracies = [
    98.25,
    97.37,
    95.61
]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies)

plt.title("ML Model Accuracy Comparison")
plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.bar_label(bars, fmt="%.2f%%")
plt.tight_layout()
plt.show()

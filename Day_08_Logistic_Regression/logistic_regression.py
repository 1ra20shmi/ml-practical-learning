
# Day 8: Logistic Regression - Breast Cancer Classification

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Target classes:", data.target_names)

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Create the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nLogistic Regression Results")
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

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Graph 2: Actual vs Predicted Classes
plt.figure(figsize=(8, 5))

sample_numbers = range(len(y_test))

plt.scatter(sample_numbers, y_test, label="Actual", marker="o")
plt.scatter(sample_numbers, y_pred, label="Predicted", marker="x")

plt.xlabel("Test Sample Number")
plt.ylabel("Class (0 = Malignant, 1 = Benign)")
plt.title("Actual vs Predicted Classes")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

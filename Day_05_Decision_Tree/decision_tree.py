from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=42)

print("Decision Tree model created successfully!")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load dataset
df = pd.read_csv("../Day_02_Dataset_Exploration/housing.csv")

# Separate features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# One-hot encode categorical data
X = pd.get_dummies(X, columns=["ocean_proximity"], dtype=int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fill missing values using training median
median_value = X_train["total_bedrooms"].median()

X_train["total_bedrooms"] = X_train["total_bedrooms"].fillna(median_value)
X_test["total_bedrooms"] = X_test["total_bedrooms"].fillna(median_value)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Create model
model = DecisionTreeRegressor(random_state=42)

print("Decision Tree model created successfully!")

# Train the model
model.fit(X_train, y_train)

print("Decision Tree model trained successfully!")

model = DecisionTreeRegressor(random_state=42)

print("Decision Tree model created successfully!")

# Train the model
model.fit(X_train, y_train)

print("Decision Tree model trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

print("\nFirst 5 predictions:")
print(y_pred[:5])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:")
print(mae)

print("\nRoot Mean Squared Error:")
print(rmse)

print("\nR² Score:")
print(r2)
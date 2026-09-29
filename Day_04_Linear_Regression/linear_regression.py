from sklearn.linear_model import LinearRegression

model = LinearRegression()

print("Linear Regression model created successfully!")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("../Day_02_Dataset_Exploration/housing.csv")

# Separate features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Convert categorical column to numerical
X = pd.get_dummies(X, columns=["ocean_proximity"], dtype=int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle missing values
median_value = X_train["total_bedrooms"].median()

X_train["total_bedrooms"] = X_train["total_bedrooms"].fillna(median_value)
X_test["total_bedrooms"] = X_test["total_bedrooms"].fillna(median_value)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")

model.fit(X_train_scaled, y_train)
# Make predictions
y_pred = model.predict(X_test_scaled)

print("\nFirst 5 predictions:")
print(y_pred[:5])

print("\nFirst 5 actual values:")
print(y_test.iloc[:5].values)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("\nMean Absolute Error:")
print(mae)

y_pred = model.predict(X_test_scaled)

print("\nFirst 5 predictions:")
print(y_pred[:5])

print("\nFirst 5 actual values:")
print(y_test.iloc[:5].values)

mae = mean_absolute_error(y_test, y_pred)

print("\nMean Absolute Error:")
print(mae)

from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nRoot Mean Squared Error:")
print(rmse)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("\nR² Score:")
print(r2)
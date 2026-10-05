
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the dataset
data = pd.read_csv("../Day_02_Dataset_Exploration/housing.csv")

# Separate features and target
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, columns=["ocean_proximity"], dtype=int)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle missing values using training data only
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Random Forest model
model = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Random Forest model created successfully!")
print("Random Forest model trained successfully!")

print("\nFirst 5 predictions:")
print(y_pred[:5])

print("\nMean Absolute Error:")
print(mean_absolute_error(y_test, y_pred))

print("\nRoot Mean Squared Error:")
print(np.sqrt(mean_squared_error(y_test, y_pred)))

print("\nR² Score:")
print(r2_score(y_test, y_pred))
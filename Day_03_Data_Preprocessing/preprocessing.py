import pandas as pd

df = pd.read_csv("../Day_02_Dataset_Exploration/housing.csv")

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nCategorical column:")
print(X["ocean_proximity"].value_counts())


X_encoded = pd.get_dummies(X, columns=["ocean_proximity"], dtype=int)

print("\nEncoded features:")
print(X_encoded.head())

print("\nNew feature shape:")
print(X_encoded.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\nMissing values in training data:")
print(X_train.isnull().sum().sum())

print("\nMissing values in testing data:")
print(X_test.isnull().sum().sum())

median_value = X_train["total_bedrooms"].median()

X_train["total_bedrooms"] = X_train["total_bedrooms"].fillna(median_value)
X_test["total_bedrooms"] = X_test["total_bedrooms"].fillna(median_value)

print("\nMissing values after imputation:")
print("Training:", X_train.isnull().sum().sum())
print("Testing:", X_test.isnull().sum().sum())

X_train["total_bedrooms"].median()

# Calculate median from training data
median_value = X_train["total_bedrooms"].median()

# Fill missing values
X_train["total_bedrooms"] = X_train["total_bedrooms"].fillna(median_value)
X_test["total_bedrooms"] = X_test["total_bedrooms"].fillna(median_value)

# Check again
print("\nMissing values after imputation:")
print("Training:", X_train.isnull().sum().sum())
print("Testing:", X_test.isnull().sum().sum())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled training shape:")
print(X_train_scaled.shape)

print("\nScaled testing shape:")
print(X_test_scaled.shape)
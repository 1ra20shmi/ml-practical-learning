import pandas as pd
import pandas as pd

df = pd.read_csv("housing.csv")

print(df.head())
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

print("\nOcean proximity counts:")
print(df["ocean_proximity"].value_counts())

print("\nData types:")
print(df.dtypes)

print("\nRows with missing total_bedrooms:")
print(df[df["total_bedrooms"].isnull()].head())

missing_percentage = df["total_bedrooms"].isnull().mean() * 100

print("\nMissing percentage:")
print(missing_percentage)

median_bedrooms = df["total_bedrooms"].median()

print("\nMedian total bedrooms:")
print(median_bedrooms)

median_bedrooms = df["total_bedrooms"].median()

df["total_bedrooms"] = df["total_bedrooms"].fillna(median_bedrooms)

print("\nMissing values after filling:")
print(df["total_bedrooms"].isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nHouse value statistics:")
print(df["median_house_value"].describe())

import matplotlib.pyplot as plt

plt.hist(df["median_house_value"], bins=50)
plt.xlabel("Median House Value")
plt.ylabel("Number of Houses")
plt.title("Distribution of House Values")
plt.show()


plt.scatter(df["median_income"], df["median_house_value"], alpha=0.3)

plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Median Income vs House Value")

plt.show()

print("\nCorrelation with house value:")
print(df.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False))

import seaborn as sns

correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
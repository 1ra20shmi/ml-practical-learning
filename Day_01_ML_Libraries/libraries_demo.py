import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("All libraries imported successfully!")


# NumPy practice

numbers = np.array([10, 20, 30, 40, 50])

print("Numbers:", numbers)
print("Mean:", np.mean(numbers))
print("Maximum:", np.max(numbers))
print("Minimum:", np.min(numbers))
print("Sum:", np.sum(numbers))


# Pandas practice

data = {
    "Name": ["Asha", "Rahul", "Priya", "Arun"],
    "Age": [21, 22, 20, 23],
    "Marks": [85, 78, 92, 88]
}

df = pd.DataFrame(data)

print("\nStudent Data:")
print(df)

print("\nFirst 2 rows:")
print(df.head(2))

print("\nDataset Shape:")
print(df.shape)

print("\nAverage Marks:")
print(df["Marks"].mean())


# Matplotlib practice

marks = [65, 72, 80, 88, 95]
students = ["A", "B", "C", "D", "E"]

plt.plot(students, marks, marker="o")

plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Marks")

plt.show()


# Seaborn practice

data = {
    "Hours": [2, 3, 4, 5, 6, 7, 8],
    "Marks": [50, 55, 60, 68, 72, 80, 88]
}

df = pd.DataFrame(data)

sns.scatterplot(x="Hours", y="Marks", data=df)

plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")

plt.show()


# Scikit-learn practice

from sklearn.linear_model import LinearRegression

# Training data
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Make prediction
prediction = model.predict([[6]])

print("\nPrediction for 6:", prediction)
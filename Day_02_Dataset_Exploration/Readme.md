# Day 2 - Dataset Exploration

## Objective
Explore and understand a real-world housing dataset using Python libraries.

## Dataset
California Housing Dataset

## What I Practiced

- Loading a CSV dataset using Pandas
- Viewing the first 5 rows using `head()`
- Checking dataset shape using `shape`
- Checking column names using `columns`
- Understanding data types using `dtypes`
- Checking dataset information using `info()`
- Finding missing values using `isnull()`
- Calculating missing-value percentage
- Handling missing values using median imputation
- Checking duplicate rows
- Statistical analysis using `describe()`
- Counting categorical values using `value_counts()`
- Creating a histogram using Matplotlib
- Creating a scatter plot
- Understanding correlation
- Creating a correlation heatmap using Seaborn

## Dataset Information

- Rows: 20,640
- Columns: 10
- Numerical columns: 9
- Categorical columns: 1
- Missing values found: 207
- Missing column: `total_bedrooms`
- Target variable: `median_house_value`

## Key Findings

- `total_bedrooms` had 207 missing values.
- The missing values were filled using median imputation.
- `ocean_proximity` is a categorical feature.
- `median_house_value` is the target variable.
- `median_income` showed a relatively strong positive correlation with `median_house_value`.

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn

## Learning Outcome

I learned how to explore, clean, analyze, and visualize a real-world dataset as part of the machine learning preparation process.
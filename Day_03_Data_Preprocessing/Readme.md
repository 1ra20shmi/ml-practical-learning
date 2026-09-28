# Day 3 - Data Preprocessing

## Objective
Prepare the housing dataset for machine learning.

## Topics Covered
- Separating features and target
- One-hot encoding
- Train-test split
- Handling missing values
- Feature scaling

## Steps

### 1. Features and Target
Separated the dataset into:
- X → input features
- y → target (`median_house_value`)

### 2. One-Hot Encoding
Converted the `ocean_proximity` categorical column into numerical columns using one-hot encoding.

Features increased from 9 to 13.

### 3. Train-Test Split
Split the dataset into:
- Training data: 80% → 16,512 rows
- Testing data: 20% → 4,128 rows

Used `random_state=42` for reproducibility.

### 4. Missing Values
Found missing values in `total_bedrooms`.

Calculated the median using the training data and used it to fill missing values in both training and testing data.

After imputation:
- Training missing values: 0
- Testing missing values: 0

### 5. Feature Scaling
Used `StandardScaler` to scale the features.

Final shapes:
- Scaled training data: 16,512 × 13
- Scaled testing data: 4,128 × 13

## Result
The housing dataset is now preprocessed and ready for machine learning model training.

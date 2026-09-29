# Day 4 - Linear Regression

## Objective
Build and evaluate a Linear Regression model using the preprocessed housing dataset.

## Topics Covered
- Linear Regression
- Model training
- Predictions
- MAE
- RMSE
- R² Score

## Steps

### 1. Load and Preprocess Data
Used the housing dataset from Day 2 and applied the preprocessing steps learned in Day 3.

### 2. Linear Regression
Created a Linear Regression model using Scikit-learn.

### 3. Model Training
Trained the model using the training data with `fit()`.

### 4. Prediction
Used `predict()` to predict house values for the test data.

### 5. Model Evaluation

#### Mean Absolute Error (MAE)
- Result: `50670.49`
- Meaning: Average difference between actual and predicted house values.

#### Root Mean Squared Error (RMSE)
- Result: `70059.19`
- Meaning: Measures prediction error and gives more weight to larger errors.

#### R² Score
- Result: `0.6254`
- Meaning: The model explains about 62.54% of the variation in the target variable.

## Result
Successfully built and evaluated the first Linear Regression model for predicting `median_house_value`.

## Libraries Used
- Pandas
- NumPy
- Scikit-learn

## Learning Outcome
Learned how to train a Linear Regression model, make predictions, and evaluate its performance using MAE, RMSE, and R² score.
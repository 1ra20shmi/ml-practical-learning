# Day 5 - Decision Tree Regression

## Objective
Build and evaluate a Decision Tree Regression model using the housing dataset.

## Topics Covered
- Decision Tree Regression
- Model training
- Predictions
- MAE
- RMSE
- R² Score

## Steps

### 1. Load and Prepare Data
Used the same housing dataset from Day 2 and performed the required preprocessing.

### 2. Decision Tree Regression
Created a `DecisionTreeRegressor` using Scikit-learn.

### 3. Model Training
Trained the model using the training data with `fit()`.

### 4. Prediction
Used `predict()` to predict house values for the test data.

### 5. Model Evaluation

#### Mean Absolute Error (MAE)
- Result: `43631.17`

#### Root Mean Squared Error (RMSE)
- Result: `69136.03`

#### R² Score
- Result: `0.6352`
- The model explains about 63.52% of the variation in the target variable.

## Result
Successfully built and evaluated a Decision Tree Regression model for predicting `median_house_value`.

## Libraries Used
- Pandas
- NumPy
- Scikit-learn

## Learning Outcome
Learned how Decision Tree Regression works, how to train a model, make predictions, and evaluate model performance using MAE, RMSE, and R².
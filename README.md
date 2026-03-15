# housing-price-prediction-model
A complete regression pipeline for housing price prediction with preprocessing, feature engineering, model benchmarking (Linear, Ridge, Lasso, Random Forest, HistGradientBoosting), and hyperparameter tuning.
# Housing Price Prediction

An end-to-end machine learning project that predicts housing prices using structured data and regression models. The project demonstrates a full ML workflow including data exploration, preprocessing, model comparison, cross-validation, hyperparameter tuning, and final evaluation.

---

## Project Overview

This project builds a predictive model for estimating **median house values** based on housing and demographic features such as:

- Median income
- Population
- Number of rooms
- Number of bedrooms
- Housing age
- Geographic coordinates
- Ocean proximity

The goal is to compare several regression algorithms and identify the model that best predicts housing prices.

---

## Dataset

The dataset used is the **California Housing dataset**, which contains information collected from the 1990 U.S. Census.

### Target Variable
`median_house_value`

### Features
- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income
- ocean_proximity

---

## Project Pipeline

The project follows a structured machine learning workflow.

### 1. Data Exploration
- Dataset inspection
- Missing value analysis
- Duplicate detection
- Feature distributions
- Correlation analysis

### 2. Data Preprocessing

Using a **scikit-learn pipeline**:

**Numerical features**
- Median imputation
- Standard scaling

**Categorical features**
- Most frequent imputation
- One-hot encoding

---

### 3. Baseline Model

A **Linear Regression model** is trained as the baseline to establish a reference performance.

Evaluation metrics:

- RMSE
- MAE
- R²

---

### 4. Model Comparison (Cross Validation)

The following models are evaluated using **5-fold cross-validation**:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Histogram Gradient Boosting Regressor

Evaluation metrics:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

---

### 5. Hyperparameter Tuning

The best-performing model (**HistGradientBoostingRegressor**) is optimized using **GridSearchCV**.

Parameters tuned include:

- learning_rate
- max_depth
- max_leaf_nodes
- min_samples_leaf
- l2_regularization

---

### 6. Final Model Evaluation

The tuned model is evaluated on the **test dataset**.

Performance metrics:

- RMSE
- MAE
- R²

Additional diagnostics include:

- Residual vs prediction plot
- Residual distribution

---

## Example Prediction

The repository includes a helper function for predicting house prices from new input data.

```python
example_pred = predict_house_price(
    model=hgb_best,
    longitude=-122.230,
    latitude=37.880,
    housing_median_age=41,
    total_rooms=880,
    total_bedrooms=129,
    population=322,
    households=126,
    median_income=8.3252,
    ocean_proximity="NEAR BAY"
)

print(example_pred)

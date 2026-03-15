import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from sklearn.metrics import(
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

#configurations
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
sns.set_theme(style="darkgrid")

plt.rcParams.update({
    "axes.titlesize":10,
    "axes.labelsize": 11,
    "xtick.labelsize": 7,
    "xtick.labelsize": 7
})

RANDOM_STATE = 44
CSV_PATH = "housing.csv"
TARGET_COL = "median_house_value"

df = pd.read_csv(CSV_PATH)
print(df.head())
print(df.columns)
print(df.info())
print(df.describe)

num_cols = df.select_dtypes(include={np.number}).columns.to_list()
cat_cols = df.select_dtypes(include={"object"}).columns.to_list()

print("Target column:", TARGET_COL)
print("num_cols:", num_cols)
print("cat_cols:", cat_cols)

print("Number of missing values:", df.isna().sum())
for col in df.columns:
    print(df[col].value_counts().head(20))

#duplicates
duplicate_mask = df.duplicated()
num_duplicates = duplicate_mask.sum()
print(num_duplicates)
#There are 0 duplicates

print(df[num_cols].describe())
print(df[num_cols].describe().T)

#visualization
# for col in cat_cols:
#     plt.figure(figsize=(10,3))
#     sns.countplot(x=col, data=df)
#     plt.title(f"Distribution of {col}")
#     plt.show()

# plt.figure(figsize=(5,8))
# sns.histplot(df[TARGET_COL], bins=40, kde=True)
# plt.title("A histogram of median house prices")
# plt.xlabel("Median House Values")
# plt.show()

# fig, axes = plt.subplots(3, 3, figsize=(8, 6))
# axes = axes.flatten()

# for i, col in enumerate(num_cols):
#     sns.histplot(df[col], kde=True, ax=axes[i])
#     axes[i].set_title(col, fontsize=8)

# plt.tight_layout()
# plt.show()

# #outlier analyis
# fig, axes = plt.subplots(3,3, figsize=(8,6))
# axes = axes.flatten()

# for i,col in enumerate(num_cols):
#     sns.boxplot(df[col], ax=axes[i])
#     axes[i].set_title(col, fontsize=8)
#     axes[i].set_xlabel("")

# plt.tight_layout()
# plt.show()

# # identify presence of highly correlated columns & feature relationships
# plt.figure(figsize=(10, 5))
# sns.heatmap(
#     df[num_cols].corr(),
#     annot=True,
#     cmap="coolwarm",
#     center=0
# )
# plt.title("Correlation Heatmap")
# plt.show()

corr_with_target = df[num_cols].corr()[TARGET_COL].sort_values(ascending=False)
print("\nCorrelation with target:")
print(corr_with_target)

x=df.drop(columns=TARGET_COL)
y=df[TARGET_COL]

print(x.head())
print(y.head())

#train test split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print("Train shape", x_train.shape)
print("Test shape:", x_test.shape)

#preprocessing pipeline
numerical_features = x_train.select_dtypes(include={np.number}).columns
categorical_features = x_train.select_dtypes(exclude={np.number}).columns

print("Numerical Features:", numerical_features)
print("Categorical features:", categorical_features)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

#Baseline Model(No CV, No tuning)
baseline_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LinearRegression())
    ]
)

#Preprocess the data and train the baseline model
baseline_pipe.fit(x_train,y_train)

#Evaluation of Baseline Model
train_baseline_pred = baseline_pipe.predict(x_train)
test_baseline_pred = baseline_pipe.predict(x_test)

train_baseline_rmse = root_mean_squared_error(y_train, train_baseline_pred)
train_baseline_mae = mean_absolute_error(y_train, train_baseline_pred)
train_baseline_r2 = r2_score(y_train, train_baseline_pred)

print("\n=== TRAIN BASELINE METRICS (LinearRegression) ===")
print(f"RMSE: {train_baseline_rmse:.3f}")
print(f"MAE : {train_baseline_mae:.3f}")
print(f"R2  : {train_baseline_r2:.3f}")

test_baseline_rmse = root_mean_squared_error(y_test, test_baseline_pred)
test_baseline_mae = mean_absolute_error(y_test, test_baseline_pred)
test_baseline_r2 = r2_score(y_test, test_baseline_pred)

print("\n=== TEST BASELINE METRICS (LinearRegression) ===")
print(f"RMSE: {test_baseline_rmse:.3f}")
print(f"MAE : {test_baseline_mae:.3f}")
print(f"R2  : {test_baseline_r2:.3f}")

#MODEL SELECTION AND OPTIMIZATION
#models to try
models ={
    "LinearRegression": LinearRegression(),
    "Ridge":Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=1000),
    "Randomforest": RandomForestRegressor(),
    "HistGB": HistGradientBoostingRegressor()
}

k = 5
cv = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

scoring={
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2"
}

rows = []
for name, model in models.items():
    pipe = Pipeline(
        steps = [
            ("preprocess", preprocess),
            ("model", model)
        ]

    )
    scores = cross_validate(pipe, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    rows.append({
        "cv_rmse": -scores["test_rmse"].mean(),
        "cv_mae": -scores["test_mae"].mean(),
        "cv_r2": scores["test_r2"].mean()    
    })

# sort based on lowest rmse value
cv_results = pd.DataFrame(rows).sort_values("cv_rmse")
print("=== CV Model Comparison ===")
print(cv_results)

#Best model is HistGB

#Hyperparameter Tuning
hgb_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE))
    ]
)

#hyperparameters combination
param_grid = {
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [None, 3, 6],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__min_samples_leaf": [20, 50, 100],
    "model__l2_regularization": [0.0, 0.1, 1.0]
}

# perform grid search
grid = GridSearchCV(
    estimator=hgb_pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose = 1
)
grid.fit(x_train, y_train)

print("\n=== TUNED HistGB (CV) ===")
print("Best CV RMSE:", -grid.best_score_)
print("Best params:", grid.best_params_)

#Retraining with Best Parameters
hgb_best = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", HistGradientBoostingRegressor(
            l2_regularization=0.1,
            learning_rate=0.1,
            max_depth=None,
            max_leaf_nodes=63,
            min_samples_leaf=20
        ))

    ]
)

hgb_best.fit(x_train, y_train)

#Final Evaluation
train_final_pred = hgb_best.predict(x_train)

train_final_rmse = root_mean_squared_error(y_train, train_final_pred)
train_final_mae = mean_absolute_error(y_train, train_final_pred)
train_final_r2 = r2_score(y_train, train_final_pred)


print("\n=== FINAL MODEL (Tuned HGB) Train Performance ===")
print(f"RMSE: {train_final_rmse:.3f}")
print(f"MAE : {train_final_mae:.3f}")
print(f"R2  : {train_final_r2:.3f}")

test_final_pred = hgb_best.predict(x_test)

test_final_rmse = root_mean_squared_error(y_test, test_final_pred)
test_final_mae = mean_absolute_error(y_test, test_final_pred)
test_final_r2 = r2_score(y_test, test_final_pred)

print("\n=== FINAL MODEL (Tuned HGB) Test Performance ===")
print(f"RMSE: {test_final_rmse:.3f}")
print(f"MAE : {test_final_mae:.3f}")
print(f"R2  : {test_final_r2:.3f}")

# residual plot
residuals = y_test - test_final_pred

plt.figure(figsize=(6, 4))
plt.scatter(test_final_pred, residuals, s=10)
plt.axhline(0)
plt.title("Residuals vs Predictions")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=40, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.show()

#Prediction
def predict_house_price(
    model,
    longitude: float,
    latitude: float,
    housing_median_age: float,
    total_rooms: float,
    total_bedrooms: float,
    population: float,
    households: float,
    median_income: float,
    ocean_proximity: str
) -> float:
    """
    Predict median_house_value for one new house.
    total_bedrooms can be np.nan (pipeline will impute).
    """
    new_row = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    return float(model.predict(new_row)[0])

#Example

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

print("\nExample prediction:", round(example_pred, 2))
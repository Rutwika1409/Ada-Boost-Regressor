import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv(
    "data/cleaned_housing.csv"
)

X = df.drop(
    "median_house_value",
    axis=1
)

y = df["median_house_value"]

X = pd.get_dummies(
    X,
    columns=["ocean_proximity"],
    drop_first=True
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(),
    random_state=42
)

param_dist = {
    "n_estimators": [50, 100, 150, 200, 300, 500],

    "learning_rate": [
        0.01,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0
    ],

    "loss": [
        "linear",
        "square",
        "exponential"
    ],

    "estimator__max_depth": [
        2,
        3,
        4,
        5,
        6,
        8
    ],

    "estimator__min_samples_split": [
        2,
        5,
        10,
        20
    ],

    "estimator__min_samples_leaf": [
        1,
        2,
        4,
        8
    ]
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=30,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    random_state=42,
    verbose=2
)

search.fit(
    X_train,
    y_train
)

best_model = search.best_estimator_

pred = best_model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    pred
)

mse = mean_squared_error(
    y_test,
    pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    pred
)

print("\nBest Parameters:")
print(search.best_params_)

print("\nMAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))

joblib.dump(
    best_model,
    "ada_boost_house_price.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "model_columns.pkl"
)

print("\nModel saved successfully.")
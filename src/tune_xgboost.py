import joblib
import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

X_train = np.load("processed/X_train.npy")
X_test = np.load("processed/X_test.npy")

y_train = np.load("processed/y_train.npy")
y_test = np.load("processed/y_test.npy")

print("Training:", X_train.shape)
print("Testing :", X_test.shape)

# =====================================================
# Base Model
# =====================================================

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

# =====================================================
# Hyperparameter Search Space
# =====================================================

param_grid = {
    "n_estimators": [300, 500, 700, 1000],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1, 2, 3]
}

# =====================================================
# Random Search
# =====================================================

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("\nStarting Hyperparameter Tuning...\n")

search.fit(X_train, y_train)

print("\nBest Parameters:")
print(search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(search.best_score_)

best_model = search.best_estimator_

# =====================================================
# Test Accuracy
# =====================================================

y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\nFinal Test Accuracy:", acc)

joblib.dump(best_model, "models/best_xgboost.pkl")

print("\nBest Model Saved Successfully!")
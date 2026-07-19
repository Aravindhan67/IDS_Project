import os
import joblib
import numpy as np
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# Create Models Folder
# ============================================================

os.makedirs("models", exist_ok=True)

# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("Loading Drebin Dataset...")
print("=" * 60)

X_train = np.load("dataset/X_train.npy")
X_test = np.load("dataset/X_test.npy")

y_train = np.load("dataset/y_train.npy")
y_test = np.load("dataset/y_test.npy")

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ============================================================
# Compute Class Weight
# ============================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight = weights[1] / weights[0]

print("\nClass Weight :", class_weight)

# ============================================================
# Build XGBoost Model
# ============================================================

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    scale_pos_weight=class_weight,

    random_state=42,

    n_jobs=-1
)

# ============================================================
# Train Model
# ============================================================

print("\nTraining Started...\n")

model.fit(X_train, y_train)

print("\nTraining Completed!")

# ============================================================
# Prediction
# ============================================================

print("\nPredicting...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy:.4f}")

# ============================================================
# Save Model
# ============================================================

joblib.dump(model, "models/drebin_xgboost.pkl")

print("\nModel Saved Successfully!")

print("\nSaved Model")
print("models/drebin_xgboost.pkl")
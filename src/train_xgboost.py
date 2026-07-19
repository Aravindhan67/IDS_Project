import os
import joblib
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ==============================================
# Load Dataset
# ==============================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

X_train = np.load("processed/X_train.npy")
X_test = np.load("processed/X_test.npy")

y_train = np.load("processed/y_train.npy")
y_test = np.load("processed/y_test.npy")

print("Training:", X_train.shape)
print("Testing :", X_test.shape)

# ==============================================
# Build Model
# ==============================================

model = XGBClassifier(
    objective="multi:softprob",
    num_class=5,

    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,

    tree_method="hist",

    eval_metric="mlogloss"
)

# ==============================================
# Train
# ==============================================

print("\nTraining Started...\n")

model.fit(X_train, y_train)

print("\nTraining Completed!")

# ==============================================
# Predict
# ==============================================

print("\nPredicting...")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {acc:.4f}")

# ==============================================
# Save Model
# ==============================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/xgboost_model.pkl")

print("\nModel Saved Successfully!")
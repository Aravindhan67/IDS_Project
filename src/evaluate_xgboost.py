import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 60)
print("Loading Model...")
print("=" * 60)

model = joblib.load("models/xgboost_model.pkl")

X_test = np.load("processed/X_test.npy")
y_test = np.load("processed/y_test.npy")

print("Predicting...")

y_pred = model.predict(X_test)

print("\n========== RESULTS ==========")

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))
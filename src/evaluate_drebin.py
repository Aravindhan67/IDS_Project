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

model = joblib.load("models/drebin_xgboost.pkl")

X_test = np.load("dataset/X_test.npy")
y_test = np.load("dataset/y_test.npy")

print("Predicting...\n")

y_pred = model.predict(X_test)

print("=" * 20, "RESULTS", "=" * 20)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malware"]
    )
)
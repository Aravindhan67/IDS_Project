import joblib
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from preprocess import preprocess

print("=" * 60)
print("Loading Model...")
print("=" * 60)

# Load trained model
model = load_model("models/cnn_model.keras")

# Load saved scaler
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/features.pkl")

# Load and preprocess test dataset
X_test, y_test, _, _ = preprocess(
    "dataset/KDDTest+.txt",
    scaler=scaler,
    feature_columns=feature_columns
)

print("Test Dataset Shape:", X_test.shape)

# Predict
y_prob = model.predict(X_test)

y_pred = (y_prob > 0.45).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== RESULTS ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report\n")
report = classification_report(y_test, y_pred)
print(report)

# Save Classification Report
with open("output/classification_report.txt", "w") as f:
    f.write(report)

# Save Metrics
with open("output/metrics.txt", "w") as f:
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")

# Save Confusion Matrix Image
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("CNN Confusion Matrix")

plt.savefig("output/confusion_matrix.png")

plt.show()

print("\nFiles Saved Successfully!")
print("output/classification_report.txt")
print("output/metrics.txt")
print("output/confusion_matrix.png")
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ============================================================
# Create Folders
# ============================================================

os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("Loading Drebin Dataset...")
print("=" * 60)

df = pd.read_csv("dataset/drebin.csv")

print("Dataset Shape :", df.shape)

# ============================================================
# Remove Missing Labels
# ============================================================

df = df.dropna(subset=["class"])

# ============================================================
# Convert Labels
# ============================================================

label_map = {
    "B": "Benign",
    "S": "Malware"
}

df["class"] = df["class"].map(label_map)

# Remove rows that could not be mapped
df = df.dropna(subset=["class"])

# ============================================================
# Features & Labels
# ============================================================

X = df.drop("class", axis=1)
y = df["class"]

# ============================================================
# Replace '?' with NaN
# ============================================================

X = X.replace("?", np.nan)

# ============================================================
# Convert Everything to Numeric
# ============================================================

X = X.apply(pd.to_numeric, errors="coerce")

# ============================================================
# Fill Missing Values
# ============================================================

X = X.fillna(0)

# ============================================================
# Convert to float32
# ============================================================

X = X.astype(np.float32)

# ============================================================
# Dataset Information
# ============================================================

print("\nNumber of Features :", X.shape[1])

print("\nMissing Values :", X.isna().sum().sum())

print("\nData Type :", X.dtypes.unique())

print("\nLabel Distribution")
print(df["class"].value_counts())

# ============================================================
# Encode Labels
# ============================================================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

print("\nClasses")

for i, c in enumerate(encoder.classes_):
    print(f"{i} -> {c}")

# ============================================================
# Normalize Features
# ============================================================

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ============================================================
# Save Files
# ============================================================

np.save("dataset/X_train.npy", X_train)
np.save("dataset/X_test.npy", X_test)

np.save("dataset/y_train.npy", y_train)
np.save("dataset/y_test.npy", y_test)

joblib.dump(scaler, "models/drebin_scaler.pkl")
joblib.dump(encoder, "models/drebin_label_encoder.pkl")

# ============================================================
# Success Message
# ============================================================

print("\n" + "=" * 60)
print("Everything Saved Successfully!")
print("=" * 60)

print("\nSaved Files")

print("dataset/X_train.npy")
print("dataset/X_test.npy")
print("dataset/y_train.npy")
print("dataset/y_test.npy")

print("models/drebin_scaler.pkl")
print("models/drebin_label_encoder.pkl")
import pandas as pd
import numpy as np

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(
    "dataset/merged_dataset.csv",
    low_memory=False
)

df.columns = df.columns.str.strip()

print("Original Shape :", df.shape)

# ---------------------------------
# Remove rows with missing label
# ---------------------------------

df = df.dropna(subset=["Label"])

print("After removing missing labels :", df.shape)

# ---------------------------------
# Replace Infinity values
# ---------------------------------

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ---------------------------------
# Remove Constant Features
# ---------------------------------

constant_features = [
    "ECE Flag Count",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate"
]

existing = [c for c in constant_features if c in df.columns]

df.drop(columns=existing, inplace=True)

print("Removed Constant Features :", len(existing))

# ---------------------------------
# Fill Missing Numeric Values
# ---------------------------------

numeric = df.select_dtypes(include=np.number).columns

df[numeric] = df[numeric].fillna(
    df[numeric].median()
)

print("Missing Values Remaining :")
print(df.isnull().sum().sum())

# ---------------------------------
# Save Clean Dataset
# ---------------------------------

df.to_csv(
    "dataset/cleaned_dataset.csv",
    index=False
)

print("\nSaved Successfully")
print("dataset/cleaned_dataset.csv")

print("Final Shape :", df.shape)
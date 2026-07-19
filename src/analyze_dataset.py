import pandas as pd
import numpy as np

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(
    "dataset/merged_dataset.csv",
    low_memory=False
)

# Remove spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset Shape:", df.shape)

# ===============================
# Missing Values
# ===============================

print("\n" + "=" * 60)
print("Missing Values")
print("=" * 60)

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No Missing Values")
else:
    print(missing)

# ===============================
# Duplicate Rows
# ===============================

print("\n" + "=" * 60)
print("Duplicate Rows")
print("=" * 60)

duplicates = df.duplicated().sum()

print("Duplicate Rows:", duplicates)

# ===============================
# Data Types
# ===============================

print("\n" + "=" * 60)
print("Data Types")
print("=" * 60)

print(df.dtypes)

# ===============================
# Numeric Features
# ===============================

numeric = df.select_dtypes(include=np.number)

print("\nNumeric Features:", numeric.shape[1])

# ===============================
# Constant Features
# ===============================

print("\n" + "=" * 60)
print("Constant Features")
print("=" * 60)

constant = []

for col in numeric.columns:
    if numeric[col].nunique() == 1:
        constant.append(col)

print("Count:", len(constant))

if constant:
    print(constant)

# ===============================
# Highly Correlated Features
# ===============================

print("\n" + "=" * 60)
print("Highly Correlated Features (>0.95)")
print("=" * 60)

corr = numeric.corr().abs()

upper = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
)

high_corr = [
    column
    for column in upper.columns
    if any(upper[column] > 0.95)
]

print("Count:", len(high_corr))

print(high_corr)

# ===============================
# Label Distribution
# ===============================

print("\n" + "=" * 60)
print("Label Distribution")
print("=" * 60)

print(df["Label"].value_counts())

print("\nAnalysis Completed Successfully!")
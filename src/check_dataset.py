import numpy as np

X = np.load("dataset/X_train.npy")
y = np.load("dataset/y_train.npy")

print("=" * 50)
print("Dataset Information")
print("=" * 50)

print("Shape:", X.shape)
print("Min:", X.min())
print("Max:", X.max())
print("Mean:", X.mean())
print("Std:", X.std())
print("NaN:", np.isnan(X).sum())
print("Inf:", np.isinf(X).sum())

print("\nLabel Distribution:")
labels, counts = np.unique(y, return_counts=True)

for label, count in zip(labels, counts):
    print(f"Class {label}: {count}")
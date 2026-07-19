import json
import os
import pandas as pd

print("=" * 60)
print("Loading Drebin Dataset...")
print("=" * 60)

df = pd.read_csv("dataset/drebin.csv")

# Remove label column
features = list(df.drop(columns=["class"]).columns)

print(f"Total Features : {len(features)}")

os.makedirs("models", exist_ok=True)

with open("models/drebin_features.json", "w") as f:
    json.dump(features, f, indent=4)

print("\nFeature list exported successfully!")

print("\nSaved File:")
print("models/drebin_features.json")
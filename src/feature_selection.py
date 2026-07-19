import joblib
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import plot_importance

print("=" * 60)
print("Loading XGBoost Model...")
print("=" * 60)

model = joblib.load("models/xgboost_model.pkl")

importance = model.feature_importances_

df = pd.read_csv(
    "dataset/cleaned_dataset.csv",
    low_memory=False
)

df.columns = df.columns.str.strip()

drop_columns = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label"
]

X = df.drop(columns=drop_columns, errors="ignore")

X = X.select_dtypes(include="number")

features = X.columns

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 30 Features\n")
print(feature_importance.head(30))

feature_importance.to_csv(
    "output/feature_importance.csv",
    index=False
)

plt.figure(figsize=(10,12))

plot_importance(
    model,
    max_num_features=20
)

plt.tight_layout()

plt.savefig("output/feature_importance.png")

plt.show()

print("\nSaved Successfully!")
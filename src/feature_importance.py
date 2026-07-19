import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("models/xgboost_model.pkl")

# Load dataset to get feature names
df = pd.read_csv(
    "dataset/CICMalDroid2020/CSV/feature_vectors_syscallsbinders_frequency_5_Cat.csv"
)

feature_names = df.columns[:-1]

importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("=" * 60)
print("Top 20 Important Features")
print("=" * 60)

print(feature_df.head(20))

feature_df.to_csv(
    "output/feature_importance.csv",
    index=False
)

plt.figure(figsize=(10,8))
plt.barh(
    feature_df["Feature"][:20],
    feature_df["Importance"][:20]
)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top 20 Important Features")

plt.tight_layout()

plt.savefig("output/feature_importance.png")

plt.show()
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocess import preprocess
from model import build_cnn

# Load and preprocess
X, y, scaler, feature_columns = preprocess(
    "dataset/KDDTrain+.txt"
)

print("Dataset Loaded Successfully")
print("Features:", X.shape)
print("Labels:", y.shape)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining:", X_train.shape)
print("Validation:", X_val.shape)

# Compute class weights
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

print("Class Weights:", class_weights)

# Build model
model = build_cnn(X_train.shape[1])

# Save best model
checkpoint = ModelCheckpoint(
    "models/cnn_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weights
)

# Save the scaler for later evaluation
joblib.dump(scaler, "models/scaler.pkl")

joblib.dump(feature_columns, "models/features.pkl")

print("\nTraining Completed Successfully!")
print("Model saved to models/cnn_model.keras")
print("Scaler saved to models/scaler.pkl")
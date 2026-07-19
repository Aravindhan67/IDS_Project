import os
import time
import json
import joblib
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================

# Automatically resolve paths regardless of where the script is run from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset paths
X_TRAIN_PATH = os.path.join(DATASET_DIR, "X_train.npy")
X_TEST_PATH = os.path.join(DATASET_DIR, "X_test.npy")
Y_TRAIN_PATH = os.path.join(DATASET_DIR, "y_train.npy")
Y_TEST_PATH = os.path.join(DATASET_DIR, "y_test.npy")

# Output artifact paths
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "drebin_xgboost_optuna.pkl")
PARAMS_SAVE_PATH = os.path.join(MODELS_DIR, "best_parameters.json")
STUDY_SAVE_PATH = os.path.join(MODELS_DIR, "optuna_study.pkl")


# ==============================================================================
# FUNCTIONS
# ==============================================================================

def load_data():
    """Load the preprocessed dataset from numpy binaries."""
    print("[*] Loading dataset...")
    X_train = np.load(X_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_test = np.load(Y_TEST_PATH)
    print("[*] Dataset loaded successfully.")
    print(f"    X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"    X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
    return X_train, X_test, y_train, y_test


def objective(trial, X, y):
    """
    Optuna objective function for hyperparameter tuning.
    Optimizes the F1 score using 5-fold Stratified Cross-Validation.
    """
    
    # 1. Define hyperparameter search space
    param = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "random_state": 42,
        "n_jobs": -1
    }
    
    # 2. Initialize XGBoost Classifier with suggested parameters
    model = xgb.XGBClassifier(**param)
    
    # 3. Use StratifiedKFold for robust evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 4. Calculate Cross-Validation F1 Score
    # cross_val_score evaluates the model internally on all 5 folds
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    
    # Optuna will try to maximize this return value
    return scores.mean()


def optimize_hyperparameters(X_train, y_train, n_trials=100):
    """Run Optuna study to find the best hyperparameters."""
    print(f"\n[*] Starting Optuna optimization with {n_trials} trials...")
    
    # Create study optimizing for maximum F1 score
    study = optuna.create_study(direction="maximize")
    
    # Run the optimization process
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
    
    print("\n[*] Optimization finished.")
    print(f"[*] Best Cross Validation F1: {study.best_value:.4f}")
    print("[*] Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    return study


def train_and_evaluate(best_params, X_train, y_train, X_test, y_test):
    """Train the final model using best parameters and evaluate on the test set."""
    print("\n[*] Training final model with best parameters...")
    
    # Explicitly set random state and threading for final model
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    
    start_time = time.time()
    
    # Initialize and train
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"[*] Training completed in {training_time:.2f} seconds.")
    
    print("\n[*] Evaluating on Test Set...")
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print metrics identically formatted
    print(f"    Final Test Accuracy  : {accuracy:.4f}")
    print(f"    Final Test Precision : {precision:.4f}")
    print(f"    Final Test Recall    : {recall:.4f}")
    print(f"    Final Test F1        : {f1:.4f}")
    
    print("\n[*] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n[*] Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model


def save_artifacts(model, study):
    """Save the trained model, best parameters, and optuna study."""
    print("\n[*] Saving artifacts...")
    
    # Save the final XGBoost Model
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"    Saved Model to {MODEL_SAVE_PATH}")
    
    # Save the Best Parameters JSON
    with open(PARAMS_SAVE_PATH, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"    Saved Parameters to {PARAMS_SAVE_PATH}")
    
    # Save the Optuna Study for potential resumption or analysis
    joblib.dump(study, STUDY_SAVE_PATH)
    print(f"    Saved Optuna Study to {STUDY_SAVE_PATH}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 60)
    print("XGBoost Hyperparameter Optimization using Optuna")
    print("=" * 60)
    
    overall_start = time.time()
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Optimize Hyperparameters (100 trials)
    study = optimize_hyperparameters(X_train, y_train, n_trials=100)
    
    # 3. Train Final Model & Evaluate
    final_model = train_and_evaluate(study.best_params, X_train, y_train, X_test, y_test)
    
    # 4. Save Output Artifacts
    save_artifacts(final_model, study)
    
    overall_time = time.time() - overall_start
    print(f"\n[*] Total execution time: {overall_time / 60:.2f} minutes.")
    print("=" * 60)
    print("Optimization Complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

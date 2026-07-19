import os
import json
import joblib
import numpy as np
import pandas as pd

from apk_analyzer.feature_extractor import FeatureExtractor


class MalwarePredictor:

    # Configuration for easy model switching
    MODEL_NAME = "drebin_xgboost_optuna.pkl"
    MODEL_DESC = "XGBoost + Optuna"
    MODEL_VERSION = "v2"
    MODEL_ACCURACY = "98.21%"

    def __init__(self):
        # Determine the project root dynamically
        # __file__ is backend/apk_analyzer/predictor.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        model_path = os.path.join(project_root, "models", self.MODEL_NAME)
        scaler_path = os.path.join(project_root, "models", "drebin_scaler.pkl")
        features_path = os.path.join(project_root, "models", "drebin_features.json")

        print("========================================")
        print("Loading AI Model...")
        print(f"Model: {self.MODEL_DESC}")
        print(f"Version: {self.MODEL_VERSION}")
        print(f"Accuracy: {self.MODEL_ACCURACY}")
        print("========================================")

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)

            with open(features_path, "r") as f:
                self.feature_names = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: Missing model file. Please ensure models are located in {os.path.join(project_root, 'models')}")
            print(f"Details: {e}")
            raise

    def predict(self, apk_parser, dex_parser):

        extractor = FeatureExtractor(apk_parser, dex_parser)

        feature_vector = extractor.generate_vector()

        # Convert to pandas DataFrame to prevent scaler warnings about missing feature names
        feature_df = pd.DataFrame([feature_vector], columns=self.feature_names)

        feature_scaled = self.scaler.transform(feature_df)

        prediction = self.model.predict(feature_scaled)[0]

        probability = self.model.predict_proba(feature_scaled)[0]

        confidence = float(np.max(probability) * 100)

        if prediction == 1:
            label = "Malware"
            risk = "High"
        else:
            label = "Benign"
            risk = "Safe"

        return {
            "prediction": label,
            "confidence": round(confidence, 2),
            "risk": risk,
            "model": self.MODEL_DESC,
            "model_version": self.MODEL_VERSION,
            "accuracy": self.MODEL_ACCURACY
        }
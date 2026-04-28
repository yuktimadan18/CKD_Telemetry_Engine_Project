"""
CKD Risk Inference Module
=========================
Loads the trained Random Forest model and performs real-time
CKD stage prediction from streaming patient telemetry data.
"""

import os
import numpy as np
import joblib


class RiskModel:
    """
    Loads the trained Random Forest model, scaler, and feature column
    ordering to perform real-time CKD stage prediction.

    Returns structured predictions with:
      - risk_score (float, 0–1)
      - stage (int, 0–4)
      - stage_label (str)
      - probabilities (dict, per-class)
    """

    TARGET_NAMES = {
        0: 'Healthy Kidney',
        1: 'Mild CKD (Stage 1\u20132)',
        2: 'Moderate CKD (Stage 3)',
        3: 'Severe CKD (Stage 4)',
        4: 'Kidney Failure (Stage 5)'
    }

    def __init__(self, model_path=None):
        """
        Load the trained model, scaler, and feature column order.

        Args:
            model_path: Path to the trained .pkl model file.
                        Scaler and feature columns are expected in the same directory.
        """
        base_dir = os.path.dirname(model_path) if model_path else 'models'

        # Load trained Random Forest model
        try:
            self.model = joblib.load(model_path or os.path.join(base_dir, 'random_forest_ckd.pkl'))
            print(f"[OK] Loaded trained model from {model_path}")
        except FileNotFoundError:
            print(f"[!] Model not found at {model_path}. Using heuristic fallback.")
            self.model = None

        # Load the StandardScaler fitted during training
        try:
            self.scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
        except FileNotFoundError:
            print("[!] Scaler not found. Predictions may be inaccurate.")
            self.scaler = None

        # Load feature column order (must match training order exactly)
        try:
            self.feature_cols = joblib.load(os.path.join(base_dir, 'feature_columns.pkl'))
        except FileNotFoundError:
            print("[!] Feature column list not found.")
            self.feature_cols = None

    def predict_risk(self, features):
        """
        Predict CKD stage from a feature dictionary (full patient tick).

        Args:
            features: dict containing all patient telemetry fields

        Returns:
            dict with keys: risk_score, stage, stage_label, probabilities
        """
        # Fallback heuristic if model/scaler/columns not loaded
        if self.model is None or self.feature_cols is None:
            egfr = features.get('eGFR', 90)
            base_risk = 1.0 - (egfr / 120.0)
            base_risk = max(0.0, min(1.0, base_risk))
            stage = (0 if base_risk < 0.25 else
                     1 if base_risk < 0.45 else
                     2 if base_risk < 0.65 else
                     3 if base_risk < 0.85 else 4)
            return {
                'risk_score': round(base_risk, 3),
                'stage': stage,
                'stage_label': f'{self.TARGET_NAMES.get(stage, "Unknown")} [fallback]',
                'probabilities': {}
            }

        # Construct feature vector in model's expected column order
        vector = []
        for col in self.feature_cols:
            val = features.get(col, 0)
            # Handle binary string values (Yes/No from CSV)
            if isinstance(val, str):
                val = 1 if val.lower() in ('yes', '1', 'true') else 0
            vector.append(float(val))

        X = np.array(vector).reshape(1, -1)

        # Apply the same scaling used during training
        if self.scaler:
            X = self.scaler.transform(X)

        # Predict stage and class probabilities
        stage = int(self.model.predict(X)[0])
        probabilities = self.model.predict_proba(X)[0]

        # Risk score = 1 − P(Healthy Kidney)
        risk_score = 1.0 - float(probabilities[0])

        return {
            'risk_score': round(risk_score, 3),
            'stage': stage,
            'stage_label': self.TARGET_NAMES.get(stage, 'Unknown'),
            'probabilities': {
                self.TARGET_NAMES[i]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            }
        }
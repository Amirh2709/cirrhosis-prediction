# src/features/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_features(self, df, month=1):
        """Create feature set for the specified month."""
        features = [
            f'Month {month} (Age)', f'Month {month} (ALT)',
            f'Month {month} (AST)', f'Month {month} (Bilirubin)',
            f'Month {month} (Albumin)', f'Month {month} (Platelets)',
            f'Month {month} (BMI)', f'Month {month} (Sodium)',
            f'Month {month} (Hemoglobin)'
        ]
        return df[features]

    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
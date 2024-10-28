# src/models/train_model.py
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


class ModelTrainer:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = self._create_model()

    def _create_model(self):
        if self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'xgboost':
            return XGBRegressor()
        elif self.model_type == 'svm':
            return SVR(kernel='rbf')
        elif self.model_type == 'lstm':
            return self._create_lstm_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _create_lstm_model(self):
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 9)),
            Dropout(0.2),
            LSTM(50, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def train(self, X_train, y_train):
        """Train the model."""
        if self.model_type == 'lstm':
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            self.model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
        else:
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate the model and return metrics."""
        if self.model_type == 'lstm':
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        y_pred = self.model.predict(X_test)

        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        return metrics
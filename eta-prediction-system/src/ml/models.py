import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Dict, Any
import time

class ETAPredictor:
    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the ETA prediction model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                objective='regression',
                metric='mae',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=42
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        
        # Train model
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'training_time': training_time,
            'accuracy': self._calculate_accuracy(y_test, y_pred)
        }
        
        self.is_trained = True
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[float, float]:
        """Make prediction with confidence score"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Make prediction
        start_time = time.time()
        prediction = self.model.predict(X)[0]
        inference_time = time.time() - start_time
        
        # Calculate confidence (simplified)
        confidence = min(0.95, max(0.5, 1.0 - (inference_time * 1000) / 100))
        
        return prediction, confidence
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy as percentage within 20% of actual"""
        within_threshold = np.abs(y_true - y_pred) / y_true <= 0.2
        return np.mean(within_threshold) * 100
    
    def save_model(self, path: str):
        """Save trained model"""
        joblib.dump(self.model, path)
    
    def load_model(self, path: str):
        """Load trained model"""
        self.model = joblib.load(path)
        self.is_trained = True
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Any
import joblib

class DataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = [
            'distance_km', 'hour', 'day_of_week', 'is_weekend',
            'is_rush_hour', 'origin_density', 'dest_density', 'vehicle_type'
        ]
    
    def prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to model input array"""
        feature_array = np.array([features[col] for col in self.feature_cols])
        return self.scaler.transform(feature_array.reshape(1, -1))
    
    def fit_scaler(self, training_data: pd.DataFrame):
        """Fit scaler on training data"""
        self.scaler.fit(training_data[self.feature_cols])
    
    def save_pipeline(self, path: str):
        """Save preprocessing pipeline"""
        joblib.dump({
            'scaler': self.scaler,
            'feature_cols': self.feature_cols
        }, path)
    
    def load_pipeline(self, path: str):
        """Load preprocessing pipeline"""
        pipeline_data = joblib.load(path)
        self.scaler = pipeline_data['scaler']
        self.feature_cols = pipeline_data['feature_cols']

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

class ModelMonitor:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.reference_stats = self._calculate_stats(reference_data)
        self.drift_threshold = 0.05  # p-value threshold for drift detection
    
    def _calculate_stats(self, data: pd.DataFrame) -> Dict:
        """Calculate statistical properties of the data"""
        return {
            'mean': data.mean().to_dict(),
            'std': data.std().to_dict(),
            'quantiles': data.quantile([0.25, 0.5, 0.75]).to_dict()
        }
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict[str, bool]:
        """Detect data drift using statistical tests"""
        drift_results = {}
        
        for column in self.reference_data.columns:
            if column in current_data.columns:
                # Kolmogorov-Smirnov test for distribution drift
                ks_statistic, p_value = stats.ks_2samp(
                    self.reference_data[column].dropna(),
                    current_data[column].dropna()
                )
                
                drift_results[column] = {
                    'drift_detected': p_value < self.drift_threshold,
                    'p_value': p_value,
                    'ks_statistic': ks_statistic
                }
        
        return drift_results
    
    def calculate_prediction_metrics(self, predictions: List[float], 
                                   actuals: List[float]) -> Dict[str, float]:
        """Calculate current model performance metrics"""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Accuracy within 20% threshold
        accuracy = np.mean(np.abs(predictions - actuals) / actuals <= 0.2) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'accuracy': accuracy,
            'sample_size': len(predictions)
        }
    
    def should_retrain(self, drift_results: Dict, 
                      performance_metrics: Dict[str, float]) -> bool:
        """Determine if model should be retrained"""
        # Check for significant drift in key features
        key_features = ['distance_km', 'hour', 'is_rush_hour']
        drift_detected = any(
            drift_results.get(feature, {}).get('drift_detected', False)
            for feature in key_features
        )
        
        # Check for performance degradation
        performance_degraded = performance_metrics.get('accuracy', 100) < 85
        
        return drift_detected or performance_degraded

class AutoRetrainer:
    def __init__(self, model_path: str, data_source: str):
        self.model_path = model_path
        self.data_source = data_source
        self.retrain_threshold_days = 7
        self.min_samples_for_retrain = 1000
    
    def should_trigger_retraining(self, monitor: ModelMonitor) -> bool:
        """Check if automatic retraining should be triggered"""
        # Get recent data for drift detection
        recent_data = self._get_recent_data()
        
        if len(recent_data) < self.min_samples_for_retrain:
            return False
        
        # Detect drift
        drift_results = monitor.detect_data_drift(recent_data)
        
        # Get recent performance metrics
        performance_metrics = self._get_recent_performance()
        
        return monitor.should_retrain(drift_results, performance_metrics)
    
    def _get_recent_data(self) -> pd.DataFrame:
        """Get recent data for analysis"""
        # This would connect to your data pipeline
        # For now, return placeholder
        return pd.DataFrame()
    
    def _get_recent_performance(self) -> Dict[str, float]:
        """Get recent model performance"""
        # This would connect to your metrics store
        # For now, return placeholder
        return {'accuracy': 92.0, 'mae': 45.0}
    
    def trigger_retraining(self):
        """Trigger model retraining pipeline"""
        logging.info("Triggering automatic model retraining")
        # This would trigger your ML pipeline
        # Could be a Kubernetes job, Airflow DAG, etc.
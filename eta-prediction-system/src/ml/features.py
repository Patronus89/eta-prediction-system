import numpy as np
import pandas as pd
from numba import jit
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import redis
import pickle

class FeatureEngine:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.feature_cache = {}
    
    @jit(nopython=True)
    def haversine_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Optimized haversine distance calculation"""
        R = 6371  # Earth's radius in km
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def generate_features(self, request: ETARequest) -> Dict[str, float]:
        """Generate features with caching for <100ms performance"""
        cache_key = f"features:{hash(str(request))}"
        
        # Check cache first
        cached_features = self.redis.get(cache_key)
        if cached_features:
            return pickle.loads(cached_features)
        
        features = {}
        
        # Basic geographical features
        features['distance_km'] = self.haversine_distance(
            request.origin.latitude, request.origin.longitude,
            request.destination.latitude, request.destination.longitude
        )
        
        # Time-based features
        now = request.departure_time or datetime.now()
        features['hour'] = now.hour
        features['day_of_week'] = now.weekday()
        features['is_weekend'] = int(now.weekday() >= 5)
        features['is_rush_hour'] = int(now.hour in [7, 8, 17, 18, 19])
        
        # Area-based features (simplified)
        features['origin_density'] = self._get_area_density(
            request.origin.latitude, request.origin.longitude
        )
        features['dest_density'] = self._get_area_density(
            request.destination.latitude, request.destination.longitude
        )
        
        # Vehicle type encoding
        vehicle_map = {'car': 0, 'bike': 1, 'truck': 2}
        features['vehicle_type'] = vehicle_map.get(request.vehicle_type, 0)
        
        # Cache features
        self.redis.setex(cache_key, 300, pickle.dumps(features))
        
        return features
    
    def _get_area_density(self, lat: float, lon: float) -> float:
        """Get area density (simplified - replace with real data)"""
        # This would connect to your area density service/database
        # For now, return a simple calculation
        return abs(lat) + abs(lon)  # Placeholder
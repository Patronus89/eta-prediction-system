from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml.data_models import ETARequest, ETAResponse
from src.ml.models import ETAPredictor
import time
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="ETA Prediction API",
    description="High-performance ETA prediction with <100ms latency",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model: ETAPredictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    
    # Load trained model
    model = ETAPredictor()
    try:
        model.load_model("models/artifacts/eta_model.pkl")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Training a new model...")
        # Create a simple fallback model
        import pandas as pd
        import numpy as np
        
        # Generate some quick training data
        np.random.seed(42)
        data = {
            'distance_km': np.random.exponential(10, 1000),
            'hour': np.random.randint(0, 24, 1000),
            'day_of_week': np.random.randint(0, 7, 1000),
            'is_weekend': np.random.choice([0, 1], 1000),
            'is_rush_hour': np.random.choice([0, 1], 1000),
            'origin_density': np.random.exponential(2, 1000),
            'dest_density': np.random.exponential(2, 1000),
            'vehicle_type': np.random.choice([0, 1, 2], 1000)
        }
        df = pd.DataFrame(data)
        base_time = df['distance_km'] * 60
        df['eta_seconds'] = base_time + np.random.normal(0, 30, 1000)
        
        X = df[['distance_km', 'hour', 'day_of_week', 'is_weekend',
                'is_rush_hour', 'origin_density', 'dest_density', 'vehicle_type']]
        y = df['eta_seconds']
        
        model.train(X, y)
        print("Fallback model trained")

def calculate_simple_features(request: ETARequest):
    """Calculate features for prediction"""
    # Simple haversine distance calculation
    from math import radians, cos, sin, asin, sqrt
    
    lat1, lon1 = request.origin.latitude, request.origin.longitude
    lat2, lon2 = request.destination.latitude, request.destination.longitude
    
    # Haversine formula
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance_km = 2 * asin(sqrt(a)) * 6371
    
    # Time features
    now = request.departure_time or datetime.now()
    hour = now.hour
    day_of_week = now.weekday()
    is_weekend = int(day_of_week >= 5)
    is_rush_hour = int(hour in [7, 8, 17, 18, 19])
    
    # Simple density calculation (placeholder)
    origin_density = abs(lat1) + abs(lon1)
    dest_density = abs(lat2) + abs(lon2)
    
    # Vehicle type encoding
    vehicle_map = {'car': 0, 'bike': 1, 'truck': 2}
    vehicle_type = vehicle_map.get(request.vehicle_type, 0)
    
    return [distance_km, hour, day_of_week, is_weekend, 
            is_rush_hour, origin_density, dest_density, vehicle_type]

@app.post("/predict", response_model=ETAResponse)
async def predict_eta(request: ETARequest) -> ETAResponse:
    """Predict ETA with <100ms latency requirement"""
    start_time = time.time()
    
    try:
        # Calculate features
        features = calculate_simple_features(request)
        
        # Make prediction
        import numpy as np
        X = np.array(features).reshape(1, -1)
        eta_seconds, confidence = model.predict(X)
        
        # Calculate latency
        latency = time.time() - start_time
        
        if latency * 1000 > 100:
            print(f"Warning: Latency exceeded 100ms: {latency*1000:.2f}ms")
        
        response = ETAResponse(
            estimated_time_seconds=int(max(0, eta_seconds)),
            confidence_score=confidence,
            route_distance_km=features[0],  # distance_km
            prediction_timestamp=datetime.now(),
            model_version="1.0.0"
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and model.is_trained,
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "ETA Prediction API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

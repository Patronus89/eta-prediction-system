from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import redis
import time
import logging
from typing import Dict, Any

from src.ml.data_models import ETARequest, ETAResponse
from src.ml.models import ETAPredictor
from src.ml.pipeline import DataPipeline
from src.ml.features import FeatureEngine
from src.utils.monitoring import MetricsCollector

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

# Global variables (loaded on startup)
model: ETAPredictor = None
pipeline: DataPipeline = None
feature_engine: FeatureEngine = None
redis_client: redis.Redis = None
metrics_collector: MetricsCollector = None

# Metrics
prediction_counter = Counter('eta_predictions_total', 'Total predictions made')
prediction_latency = Histogram('eta_prediction_duration_seconds', 'Prediction latency')
error_counter = Counter('eta_prediction_errors_total', 'Total prediction errors')

@app.on_event("startup")
async def startup_event():
    """Initialize models and connections on startup"""
    global model, pipeline, feature_engine, redis_client, metrics_collector
    
    # Initialize Redis
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
    
    # Load model and pipeline
    model = ETAPredictor()
    model.load_model("models/artifacts/eta_model.pkl")
    
    pipeline = DataPipeline()
    pipeline.load_pipeline("models/artifacts/pipeline.pkl")
    
    # Initialize feature engine
    feature_engine = FeatureEngine(redis_client)
    
    # Initialize metrics collector
    metrics_collector = MetricsCollector()
    
    logging.info("ETA Prediction API started successfully")

@app.post("/predict", response_model=ETAResponse)
async def predict_eta(request: ETARequest) -> ETAResponse:
    """Predict ETA with <100ms latency requirement"""
    start_time = time.time()
    
    try:
        # Generate features
        features = feature_engine.generate_features(request)
        
        # Prepare features for model
        model_input = pipeline.prepare_features(features)
        
        # Make prediction
        eta_seconds, confidence = model.predict(model_input)
        
        # Calculate total latency
        latency = time.time() - start_time
        
        # Record metrics
        prediction_counter.inc()
        prediction_latency.observe(latency)
        
        # Check latency requirement
        if latency * 1000 > 100:  # Convert to milliseconds
            logging.warning(f"Prediction latency exceeded 100ms: {latency*1000:.2f}ms")
        
        response = ETAResponse(
            estimated_time_seconds=int(eta_seconds),
            confidence_score=confidence,
            route_distance_km=features['distance_km'],
            prediction_timestamp=datetime.now(),
            model_version="1.0.0"
        )
        
        return response
        
    except Exception as e:
        error_counter.inc()
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and model.is_trained,
        "redis_connected": redis_client.ping() if redis_client else False,
        "timestamp": time.time()
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    return {
        "model_type": model.model_type if model else None,
        "version": "1.0.0",
        "features": pipeline.feature_cols if pipeline else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
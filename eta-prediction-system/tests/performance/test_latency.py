import pytest
import time
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from httpx import AsyncClient
    import httpx
except ImportError:
    pytest.skip("httpx not installed", allow_module_level=True)

@pytest.mark.asyncio
async def test_prediction_latency():
    """Test that predictions are under 100ms"""
    request_data = {
        "origin": {"latitude": 37.7749, "longitude": -122.4194},
        "destination": {"latitude": 37.7849, "longitude": -122.4094},
        "vehicle_type": "car"
    }
    
    base_url = "http://localhost:8000"
    
    # First check if API is running
    try:
        async with AsyncClient(base_url=base_url) as client:
            health_response = await client.get("/health")
            if health_response.status_code != 200:
                pytest.skip("API not running on localhost:8000")
    except:
        pytest.skip("Cannot connect to API on localhost:8000")
    
    latencies = []
    async with AsyncClient(base_url=base_url) as client:
        for _ in range(10):  # Test 10 requests
            start_time = time.time()
            response = await client.post("/predict", json=request_data)
            latency = time.time() - start_time
            latencies.append(latency * 1000)  # Convert to milliseconds
            
            assert response.status_code == 200
            assert latency < 0.1  # Less than 100ms
    
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
    
    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"P95 latency: {p95_latency:.2f}ms")
    
    assert avg_latency < 50  # Target average under 50ms
    assert p95_latency < 100  # P95 under 100ms

def test_api_health():
    """Test that API health endpoint works"""
    import requests
    
    try:
        response = requests.get("http://localhost:8000/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
    except requests.exceptions.ConnectionError:
        pytest.skip("API not running on localhost:8000")

def test_prediction_accuracy_format():
    """Test that prediction response has correct format"""
    import requests
    
    request_data = {
        "origin": {"latitude": 37.7749, "longitude": -122.4194},
        "destination": {"latitude": 37.7849, "longitude": -122.4094},
        "vehicle_type": "car"
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "estimated_time_seconds" in data
        assert "confidence_score" in data
        assert "route_distance_km" in data
        assert "prediction_timestamp" in data
        assert "model_version" in data
        
        assert isinstance(data["estimated_time_seconds"], int)
        assert 0 <= data["confidence_score"] <= 1
        assert data["route_distance_km"] > 0
        
    except requests.exceptions.ConnectionError:
        pytest.skip("API not running on localhost:8000")

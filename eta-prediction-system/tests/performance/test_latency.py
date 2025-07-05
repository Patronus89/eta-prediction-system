import pytest
import time
import asyncio
from httpx import AsyncClient
from src.api.main import app

@pytest.mark.asyncio
async def test_prediction_latency():
    """Test that predictions are under 100ms"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        request_data = {
            "origin": {"latitude": 37.7749, "longitude": -122.4194},
            "destination": {"latitude": 37.7849, "longitude": -122.4094},
            "vehicle_type": "car"
        }
        
        latencies = []
        for _ in range(100):
            start_time = time.time()
            response = await client.post("/predict", json=request_data)
            latency = time.time() - start_time
            latencies.append(latency * 1000)  # Convert to milliseconds
            
            assert response.status_code == 200
            assert latency < 0.1  # Less than 100ms
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[95]
        
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"P95 latency: {p95_latency:.2f}ms")
        
        assert avg_latency < 50  # Target average under 50ms
        assert p95_latency < 100  # P95 under 100ms
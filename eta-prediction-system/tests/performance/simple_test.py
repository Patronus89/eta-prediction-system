import requests
import time
import statistics

def test_latency(num_requests=100):
    """Simple latency test"""
    url = "http://localhost:8000/predict"
    data = {
        "origin": {"latitude": 37.7749, "longitude": -122.4194},
        "destination": {"latitude": 37.7849, "longitude": -122.4094},
        "vehicle_type": "car"
    }
    
    latencies = []
    errors = 0
    
    print(f"Testing {num_requests} sequential requests...")
    
    for i in range(num_requests):
        start_time = time.time()
        try:
            response = requests.post(url, json=data)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            if response.status_code != 200:
                errors += 1
                
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_requests} requests...")
                
        except Exception as e:
            errors += 1
            print(f"Error on request {i + 1}: {e}")
    
    if latencies:
        print(f"\n=== Results ===")
        print(f"Average latency: {statistics.mean(latencies):.2f}ms")
        print(f"Median latency: {statistics.median(latencies):.2f}ms")
        print(f"Min latency: {min(latencies):.2f}ms")
        print(f"Max latency: {max(latencies):.2f}ms")
        print(f"Errors: {errors}/{num_requests}")
        
        # Check 100ms requirement
        over_100ms = sum(1 for l in latencies if l > 100)
        print(f"Requests over 100ms: {over_100ms}/{len(latencies)} ({over_100ms/len(latencies)*100:.1f}%)")

if __name__ == "__main__":
    # First check if API is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ API is running")
            test_latency()
        else:
            print("❌ API health check failed")
    except:
        print("❌ Cannot connect to API. Make sure it's running on http://localhost:8000")
        print("Start it with: python src/api/main.py")

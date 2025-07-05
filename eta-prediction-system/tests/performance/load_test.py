import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor
import statistics
import json

async def make_request(session, url, data):
    """Make a single request"""
    start_time = time.time()
    try:
        async with session.post(url, json=data) as response:
            result = await response.json()
            latency = time.time() - start_time
            return latency, response.status, result
    except Exception as e:
        latency = time.time() - start_time
        return latency, 500, {"error": str(e)}

async def load_test(concurrent_requests=50, total_requests=500):
    """Run load test"""
    url = "http://localhost:8000/predict"
    request_data = {
        "origin": {"latitude": 37.7749, "longitude": -122.4194},
        "destination": {"latitude": 37.7849, "longitude": -122.4094},
        "vehicle_type": "car"
    }
    
    connector = aiohttp.TCPConnector(limit=concurrent_requests)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for _ in range(total_requests):
            task = make_request(session, url, request_data)
            tasks.append(task)
        
        print(f"Starting load test: {total_requests} requests with {concurrent_requests} concurrent...")
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
    
    # Analyze results
    latencies = [r[0] * 1000 for r in results]  # Convert to ms
    statuses = [r[1] for r in results]
    success_rate = sum(1 for s in statuses if s == 200) / len(statuses)
    
    print(f"\n=== Load Test Results ===")
    print(f"Total Requests: {total_requests}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Requests/second: {total_requests/total_time:.2f}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Latency: {statistics.mean(latencies):.2f}ms")
    print(f"Median Latency: {statistics.median(latencies):.2f}ms")
    print(f"P95 Latency: {sorted(latencies)[int(0.95 * len(latencies))]:.2f}ms")
    print(f"P99 Latency: {sorted(latencies)[int(0.99 * len(latencies))]:.2f}ms")
    print(f"Min Latency: {min(latencies):.2f}ms")
    print(f"Max Latency: {max(latencies):.2f}ms")
    
    # Check if requirements are met
    avg_latency = statistics.mean(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
    
    print(f"\n=== Performance Requirements ===")
    print(f"✅ Average < 50ms: {avg_latency:.2f}ms {'✅' if avg_latency < 50 else '❌'}")
    print(f"✅ P95 < 100ms: {p95_latency:.2f}ms {'✅' if p95_latency < 100 else '❌'}")
    print(f"✅ Success rate > 99%: {success_rate:.2%} {'✅' if success_rate > 0.99 else '❌'}")

if __name__ == "__main__":
    # Test if API is running first
    async def check_api():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/health") as response:
                    if response.status == 200:
                        return True
        except:
            return False
        return False
    
    async def main():
        api_running = await check_api()
        if not api_running:
            print("❌ API is not running on http://localhost:8000")
            print("Please start the API first with: python src/api/main.py")
            return
        
        print("✅ API is running, starting load test...")
        await load_test()
    
    asyncio.run(main())

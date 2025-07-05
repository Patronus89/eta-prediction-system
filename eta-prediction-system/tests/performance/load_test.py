import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor
import statistics

async def make_request(session, url, data):
    """Make a single request"""
    start_time = time.time()
    async with session.post(url, json=data) as response:
        result = await response.json()
        latency = time.time() - start_time
        return latency, response.status, result

async def load_test(concurrent_requests=100, total_requests=1000):
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
        
        results = await asyncio.gather(*tasks)
    
    latencies = [r[0] * 1000 for r in results]  # Convert to ms
    success_rate = sum(1 for r in results if r[1] == 200) / len(results)
    
    print(f"Load Test Results:")
    print(f"  Total Requests: {total_requests}")
    print(f"  Success Rate: {success_rate:.2%}")
    print(f"  Average Latency: {statistics.mean(latencies):.2f}ms")
    print(f"  Median Latency: {statistics.median(latencies):.2f}ms")
    print(f"  P95 Latency: {sorted(latencies)[int(0.95 * len(latencies))]:.2f}ms")
    print(f"  P99 Latency: {sorted(latencies)[int(0.99 * len(latencies))]:.2f}ms")

if __name__ == "__main__":
    asyncio.run(load_test())
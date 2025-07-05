import time
import logging
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge
import psutil
import threading

class MetricsCollector:
    def __init__(self):
        self.request_count = Counter('http_requests_total', 'Total HTTP requests')
        self.response_time = Histogram('http_request_duration_seconds', 'HTTP request duration')
        self.active_connections = Gauge('active_connections', 'Active connections')
        self.memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        
        # Start background monitoring
        self.start_system_monitoring()
    
    def start_system_monitoring(self):
        """Start background thread for system monitoring"""
        def monitor():
            while True:
                # Update system metrics
                self.memory_usage.set(psutil.virtual_memory().used)
                self.cpu_usage.set(psutil.cpu_percent())
                time.sleep(10)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        self.request_count.labels(method=method, endpoint=endpoint, status=status_code).inc()
        self.response_time.labels(method=method, endpoint=endpoint).observe(duration)
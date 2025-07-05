import redis
import pickle
import hashlib
from typing import Any, Optional
import time

class CacheManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 300  # 5 minutes
    
    def get_cache_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data"""
        data_str = str(data)
        hash_obj = hashlib.md5(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            cached_data = self.redis.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception:
            pass
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set cached value"""
        try:
            ttl = ttl or self.default_ttl
            serialized_data = pickle.dumps(value)
            self.redis.setex(key, ttl, serialized_data)
            return True
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cached value"""
        try:
            self.redis.delete(key)
            return True
        except Exception:
            return False
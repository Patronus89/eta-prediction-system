import random
import numpy as np
from typing import Dict, List, Optional
import redis
import json
from datetime import datetime

class ABTestManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.experiments = {}
    
    def create_experiment(self, experiment_id: str, 
                         model_a_path: str, model_b_path: str,
                         traffic_split: float = 0.5) -> bool:
        """Create a new A/B test experiment"""
        experiment_config = {
            'experiment_id': experiment_id,
            'model_a_path': model_a_path,
            'model_b_path': model_b_path,
            'traffic_split': traffic_split,
            'start_time': datetime.now().isoformat(),
            'status': 'active',
            'metrics': {
                'model_a': {'requests': 0, 'total_latency': 0, 'errors': 0},
                'model_b': {'requests': 0, 'total_latency': 0, 'errors': 0}
            }
        }
        
        # Store experiment config in Redis
        self.redis.set(
            f"experiment:{experiment_id}",
            json.dumps(experiment_config)
        )
        
        return True
    
    def get_model_for_request(self, experiment_id: str, user_id: str) -> str:
        """Determine which model to use for a request"""
        # Get experiment config
        experiment_data = self.redis.get(f"experiment:{experiment_id}")
        if not experiment_data:
            return "model_a"  # Default to model A
        
        experiment = json.loads(experiment_data)
        
        # Consistent assignment based on user_id
        hash_value = hash(user_id) % 100
        if hash_value < experiment['traffic_split'] * 100:
            return "model_a"
        else:
            return "model_b"
    
    def record_result(self, experiment_id: str, model_version: str,
                     latency: float, error: bool = False):
        """Record experiment results"""
        experiment_data = self.redis.get(f"experiment:{experiment_id}")
        if not experiment_data:
            return
        
        experiment = json.loads(experiment_data)
        
        # Update metrics
        metrics = experiment['metrics'][model_version]
        metrics['requests'] += 1
        metrics['total_latency'] += latency
        if error:
            metrics['errors'] += 1
        
        # Save updated experiment
        self.redis.set(
            f"experiment:{experiment_id}",
            json.dumps(experiment)
        )
    
    def get_experiment_results(self, experiment_id: str) -> Dict:
        """Get current experiment results"""
        experiment_data = self.redis.get(f"experiment:{experiment_id}")
        if not experiment_data:
            return {}
        
        experiment = json.loads(experiment_data)
        metrics = experiment['metrics']
        
        results = {}
        for model in ['model_a', 'model_b']:
            model_metrics = metrics[model]
            requests = model_metrics['requests']
            
            if requests > 0:
                results[model] = {
                    'requests': requests,
                    'avg_latency': model_metrics['total_latency'] / requests,
                    'error_rate': model_metrics['errors'] / requests,
                    'success_rate': 1 - (model_metrics['errors'] / requests)
                }
        
        return results
    
    def determine_winner(self, experiment_id: str, 
                        min_samples: int = 1000) -> Optional[str]:
        """Determine winning model using statistical significance"""
        results = self.get_experiment_results(experiment_id)
        
        if not results or len(results) < 2:
            return None
        
        model_a_metrics = results.get('model_a', {})
        model_b_metrics = results.get('model_b', {})
        
        # Check if we have enough samples
        if (model_a_metrics.get('requests', 0) < min_samples or 
            model_b_metrics.get('requests', 0) < min_samples):
            return None
        
        # Simple comparison based on success rate and latency
        a_score = (model_a_metrics.get('success_rate', 0) * 0.6 + 
                  (1 - model_a_metrics.get('avg_latency', 1) / 1000) * 0.4)
        b_score = (model_b_metrics.get('success_rate', 0) * 0.6 + 
                  (1 - model_b_metrics.get('avg_latency', 1) / 1000) * 0.4)
        
        if abs(a_score - b_score) < 0.05:  # No significant difference
            return None
        
        return 'model_a' if a_score > b_score else 'model_b'
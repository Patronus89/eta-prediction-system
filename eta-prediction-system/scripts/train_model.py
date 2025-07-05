import pandas as pd
import numpy as np
from src.ml.models import ETAPredictor
from src.ml.pipeline import DataPipeline
from src.ml.features import FeatureEngine
import argparse
import yaml
import logging

def generate_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic training data"""
    np.random.seed(42)
    
    data = {
        'distance_km': np.random.exponential(10, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'is_rush_hour': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'origin_density': np.random.exponential(2, n_samples),
        'dest_density': np.random.exponential(2, n_samples),
        'vehicle_type': np.random.choice([0, 1, 2], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate target (ETA in seconds) with realistic relationships
    base_time = df['distance_km'] * 60  # 1 minute per km base
    rush_multiplier = 1 + df['is_rush_hour'] * 0.5
    weekend_multiplier = 1 - df['is_weekend'] * 0.1
    density_effect = 1 + (df['origin_density'] + df['dest_density']) * 0.1
    
    df['eta_seconds'] = (base_time * rush_multiplier * 
                        weekend_multiplier * density_effect +
                        np.random.normal(0, 30, n_samples))
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--model-type', default='lightgbm')
    parser.add_argument('--output-dir', default='models/artifacts')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate training data
    print("Generating synthetic training data...")
    df = generate_synthetic_data(50000)
    
    # Prepare features and target
    feature_cols = [
        'distance_km', 'hour', 'day_of_week', 'is_weekend',
        'is_rush_hour', 'origin_density', 'dest_density', 'vehicle_type'
    ]
    
    X = df[feature_cols]
    y = df['eta_seconds']
    
    # Initialize pipeline and fit scaler
    pipeline = DataPipeline()
    pipeline.fit_scaler(X)
    
    # Train model
    print(f"Training {args.model_type} model...")
    model = ETAPredictor(model_type=args.model_type)
    metrics = model.train(X, y)
    
    print(f"Training Results:")
    print(f"  MAE: {metrics['mae']:.2f} seconds")
    print(f"  RMSE: {metrics['rmse']:.2f} seconds")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Training Time: {metrics['training_time']:.2f} seconds")
    
    # Save model and pipeline
    model.save_model(f"{args.output_dir}/eta_model.pkl")
    pipeline.save_pipeline(f"{args.output_dir}/pipeline.pkl")
    
    print(f"Model saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
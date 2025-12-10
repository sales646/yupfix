import os
import sys
import logging

# Ensure project root is on PYTHONPATH for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
# Explicitly add the 'env' package path
sys.path.append(os.path.join(PROJECT_ROOT, 'env'))

from stable_baselines3 import PPO
from ftmo_env import FTMOEvaluator  # ftmo_env.py resides in env folder
from src.data.dukascopy_loader import DukascopyLoader
from src.data.feature_engineer import FeatureEngineer
from callbacks.early_stopping import EarlyStoppingCallback
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CurriculumTrainer")

def train_stage(start_date: str, end_date: str, profit_target: float, max_loss_limit: float, timesteps: int, model_path: str = None):
    """Train a PPO agent for a specific profit target.

    Parameters
    ----------
    start_date: str
        Start date for historical data (YYYY-MM-DD).
    end_date: str
        End date for historical data (YYYY-MM-DD).
    profit_target: float
        Desired profit target (e.g., 0.01 for 1%).
    max_loss_limit: float
        Maximum allowed loss as a fraction of equity.
    timesteps: int
        Number of training timesteps.
    model_path: str, optional
        Path to a previously saved PPO model to warm‑start from.
    """
    # Load price data from existing parquet file
    parquet_file = "data/raw/EURUSD_1m_2023-11-23_2025-11-22.parquet"
    logger.info(f"Loading data from {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    # Filter to date range
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    if df.empty:
        raise RuntimeError(f"No data found in range {start_date} to {end_date}")
    
    logger.info(f"Loaded {len(df)} rows from parquet file")
    
    # Feature engineering
    fe = FeatureEngineer()
    df = fe.calculate_features(df)
    df = df.dropna().reset_index(drop=False)

    # Create the FTMO environment with the given targets
    env = FTMOEvaluator(df, profit_target=profit_target, max_loss_limit=max_loss_limit, symbol="EURUSD")

    # Load or initialise PPO model
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading pretrained model from {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        model = PPO("MlpPolicy", env, verbose=1)

    # Create early stopping callback
    # Relaxed settings to avoid premature stopping
    early_stop = EarlyStoppingCallback(patience=15, min_delta=0.05, check_freq=10000, verbose=1)

    logger.info(f"Training stage: profit_target={profit_target*100:.2f}% for {timesteps} timesteps")
    model.learn(total_timesteps=timesteps, callback=early_stop)
    return model

if __name__ == "__main__":
    # Define curriculum stages (profit target, timesteps)
    stages = [
        {"name": "stage1", "target": 0.01, "timesteps": 50000},   # +1%
        {"name": "stage2", "target": 0.05, "timesteps": 75000},   # +5%
        {"name": "stage3", "target": 0.10, "timesteps": 100000},  # +10%
    ]
    # Common data window for all stages (using existing 2024 data)
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    max_loss = 0.10
    prev_path = None
    for s in stages:
        model = train_stage(start_date, end_date, s["target"], max_loss, s["timesteps"], model_path=prev_path)
        out_path = f"models/ppo_{s['name']}.zip"
        model.save(out_path)
        logger.info(f"Saved {s['name']} model to {out_path}")
        prev_path = out_path
    # Final model for the system
    final_path = "models/ppo_ftmo_agent.zip"
    os.replace(prev_path, final_path)
    logger.info(f"Final PPO model ready at {final_path}")

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.data.dukascopy_loader import DukascopyLoader
from src.data.feature_engineer import FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExpertGen")

def generate_expert_data(symbol="EURUSD", start_date="2022-01-01", end_date="2023-01-01"):
    """
    Generate (Observation, Action) pairs based on future price movement.
    Action 1 (Buy) if price increases > threshold in next N steps.
    Action 2 (Sell) if price decreases > threshold in next N steps.
    Action 0 (Hold) otherwise.
    """
    logger.info(f"Fetching data for {symbol} ({start_date}-{end_date})...")
    # Use local parquet file for speed and consistency
    parquet_file = "data/raw/EURUSD_1m_2023-11-23_2025-11-22.parquet"
    if os.path.exists(parquet_file):
        logger.info(f"Loading from {parquet_file}")
        df = pd.read_parquet(parquet_file)
        # Filter date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
    else:
        logger.warning("Parquet not found, downloading...")
        loader = DukascopyLoader()
        df = loader.fetch_data(symbol, start_date, end_date, resample="1min")
    
    logger.info("Calculating features...")
    fe = FeatureEngineer()
    df = fe.calculate_features(df)
    df = df.dropna().reset_index(drop=False) # Ensure index is reset for iteration
    
    # Parameters for "Expert" Logic
    LOOKAHEAD_STEPS = 60 # 1 hour
    PROFIT_THRESHOLD = 0.0010 # 10 pips (approx 0.1%)
    
    observations = []
    actions = []
    
    logger.info("Labeling data...")
    
    for i in range(len(df) - LOOKAHEAD_STEPS):
        current_price = df.iloc[i]['close']
        future_price = df.iloc[i + LOOKAHEAD_STEPS]['close']
        
        # Calculate return
        ret = (future_price - current_price) / current_price
        
        action = 0 # Hold
        if ret > PROFIT_THRESHOLD:
            action = 1 # Buy
        elif ret < -PROFIT_THRESHOLD:
            action = 2 # Sell
            
        # Extract observation (must match Environment observation space!)
        # [ret_1, ret_5, garch_vol, rsi, macd, dist_ma, dd_pct, ml_signal, atr, hour_sin, hour_cos, session]
        
        row = df.iloc[i]
        obs = [
            row.get('ret_1', 0),
            row.get('ret_5', 0),
            row.get('garch_vol', 0),
            row.get('rsi', 50),
            row.get('macd', 0),
            row.get('dist_ma_20', 0),
            0.0, # Mock Drawdown
            0.5, # Mock ML Signal
            # New Features
            row.get('atr', 0),
            row.get('hour_sin', 0),
            row.get('hour_cos', 0),
            row.get('session', 0)
        ]
        
        observations.append(obs)
        actions.append(action)
        
        if i % 10000 == 0:
            logger.info(f"Processed {i}/{len(df)} steps")
            
    # Save to file
    obs_array = np.array(observations, dtype=np.float32)
    act_array = np.array(actions, dtype=np.int64)
    
    np.save("data/expert_obs.npy", obs_array)
    np.save("data/expert_actions.npy", act_array)
    
    logger.info(f"Saved {len(obs_array)} samples to data/expert_*.npy")
    logger.info(f"Class Balance: Hold={np.sum(act_array==0)}, Buy={np.sum(act_array==1)}, Sell={np.sum(act_array==2)}")

if __name__ == "__main__":
    # Create data dir if not exists
    os.makedirs("data", exist_ok=True)
    # Use 2024 data for expert generation (matching PPO training data)
    generate_expert_data(start_date="2024-01-01", end_date="2024-12-31")

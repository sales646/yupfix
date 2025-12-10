import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from src.models.rl.environment import FTMOEvaluator
from src.models.supervised.xgb_model import XGBoostModel
from src.data.dukascopy_loader import DukascopyLoader
from joblib import Parallel, delayed
import logging
import itertools

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Tuner")

def run_episode(seed, data_pool, threshold, risk_per_trade):
    """
    Run a single simulation episode with specific parameters.
    """
    np.random.seed(seed)
    
    if len(data_pool) < 30 * 1440:
        return "ERROR_NO_DATA"
        
    start_idx = np.random.randint(0, len(data_pool) - (30 * 1440))
    df_slice = data_pool.iloc[start_idx : start_idx + (30 * 1440)].copy()
    
    # Recalculate signals based on new threshold
    # We assume probs are in df_slice columns 'prob_down', 'prob_neutral', 'prob_up'
    # (We need to ensure prepare_data saves these)
    
    actions = np.ones(len(df_slice), dtype=int) # Default 1 (Hold)
    actions[df_slice['prob_up'] > threshold] = 2 # Buy
    actions[df_slice['prob_down'] > threshold] = 0 # Sell
    
    df_slice['signal'] = actions
    
    # Initialize Environment with custom risk
    # Note: FTMOEvaluator might need update to accept risk_per_trade if it's hardcoded
    # For now, we'll assume standard 1 lot or modify env later. 
    # Actually, let's just simulate PnL scaling here for simplicity if Env doesn't support it.
    # But Env uses fixed logic. Let's stick to threshold tuning first, 
    # as risk sizing requires Env changes.
    
    env = FTMOEvaluator(df_slice)
    obs, _ = env.reset()
    done = False
    
    while not done:
        current_idx = env.current_step
        if current_idx >= len(df_slice):
            break
            
        action = df_slice.iloc[current_idx]['signal']
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if terminated:
            if env.equity >= env.initial_balance * (1 + env.profit_target):
                return "PASS"
            elif env.equity <= env.initial_balance * (1 - env.max_loss_limit):
                return "FAIL_MAX_LOSS"
            elif (env.start_of_day_equity - env.equity) / env.start_of_day_equity >= env.daily_loss_limit:
                return "FAIL_DAILY_LOSS"
                
    return "TIMEOUT"

def prepare_data_with_probs():
    """
    Load data and pre-calculate raw probabilities.
    """
    logger.info("Loading validation data (2023-2024)...")
    loader = DukascopyLoader()
    df = loader.fetch_data("EURUSD", "2023-01-01", "2023-04-01", resample="1min")
    
    if df is None or df.empty:
        return None

    logger.info("Loading trained model...")
    model = XGBoostModel()
    try:
        model.load()
    except:
        return None
        
    logger.info("Generating probabilities...")
    df_features = model.prepare_features(df)
    
    import xgboost as xgb
    dtest = xgb.DMatrix(df_features)
    probs = model.model.predict(dtest) # (N, 3)
    
    df_features['prob_down'] = probs[:, 0]
    df_features['prob_neutral'] = probs[:, 1]
    df_features['prob_up'] = probs[:, 2]
    
    # Join EVERYTHING (features + probs)
    full_data = df.join(df_features, how='inner', lsuffix='_orig')
    return full_data

def main():
    print("Preparing Data...")
    data_pool = prepare_data_with_probs()
    if data_pool is None:
        print("❌ Error loading data")
        return

    # Grid Search
    thresholds = [0.50, 0.55, 0.60, 0.65]
    # risk_per_trade = [0.005, 0.01] # Not implemented in Env yet
    
    print(f"\n🔎 Tuning Parameters (Thresholds: {thresholds})")
    
    results_summary = []
    
    for th in thresholds:
        print(f"\nTesting Threshold: {th}")
        n_sims = 50 # Faster for tuning
        
        results = Parallel(n_jobs=-1)(delayed(run_episode)(i, data_pool, th, 0.01) for i in range(n_sims))
        
        pass_rate = results.count("PASS") / n_sims
        fail_rate = (results.count("FAIL_MAX_LOSS") + results.count("FAIL_DAILY_LOSS")) / n_sims
        survival = results.count("TIMEOUT") / n_sims
        
        print(f"  Pass: {pass_rate:.1%} | Fail: {fail_rate:.1%} | Survive: {survival:.1%}")
        results_summary.append((th, pass_rate, fail_rate))

    # Find Best
    best = max(results_summary, key=lambda x: x[1]) # Maximize Pass Rate
    print(f"\n🏆 Best Settings: Threshold={best[0]} (Pass Rate: {best[1]:.1%})")

if __name__ == "__main__":
    main()

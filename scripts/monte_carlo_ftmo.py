import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from env.ftmo_env import FTMOEvaluator
from src.data.feature_engineer import FeatureEngineer
import logging
from joblib import Parallel, delayed
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MonteCarlo")

def run_simulation(seed, df_full, agent_path):
    """
    Run a single Monte Carlo simulation episode.
    """
    np.random.seed(seed)
    
    # 1. Select Random Start Date
    # We need at least 30 days of data (approx 30 * 1440 minutes)
    min_length = 30 * 1440
    if len(df_full) < min_length:
        return {"result": "ERROR", "profit": 0, "drawdown": 0}
        
    max_start = len(df_full) - min_length
    start_idx = np.random.randint(0, max_start)
    
    # Slice data for this episode
    df_slice = df_full.iloc[start_idx : start_idx + min_length].copy().reset_index(drop=True)
    
    # 2. Initialize Environment
    # Note: We create a fresh env for each thread to avoid conflicts
    env = FTMOEvaluator(df_slice, symbol="EURUSD")
    
    # 3. Load Agent (We have to load it inside the process or share it carefully)
    # Loading it 100 times is slow, but safe for Parallel. 
    # Better: Load once and pass? PPO objects might not be pickleable for joblib easily.
    # Let's try loading inside first.
    try:
        # Suppress warnings
        import warnings
        warnings.filterwarnings("ignore")
        agent = PPO.load(agent_path, device='cpu')
    except Exception as e:
        return {"result": "ERROR_LOAD", "profit": 0, "drawdown": 0}
    
    obs, _ = env.reset()
    done = False
    
    max_drawdown = 0.0
    
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            action = int(action.item())
            
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Track Max Drawdown
        current_dd = (env.initial_balance - env.equity) / env.initial_balance
        if current_dd > max_drawdown:
            max_drawdown = current_dd
            
    # Result
    profit_pct = (env.equity - env.initial_balance) / env.initial_balance
    
    result = "TIMEOUT"
    if env.equity >= env.initial_balance * 1.10:
        result = "PASS"
    elif env.equity <= env.initial_balance * 0.90:
        result = "FAIL_MAX_LOSS"
    elif max_drawdown >= 0.05: # Approximate check, env handles daily loss
        # If env terminated due to daily loss, equity might not be -10%
        # We check termination reason via equity or just trust env
        if env.equity < env.initial_balance:
             result = "FAIL_DAILY_LOSS" # Likely cause if terminated early with loss
    
    return {
        "result": result,
        "profit": profit_pct,
        "drawdown": max_drawdown,
        "trades": env.trades_count
    }

def run_monte_carlo(n_simulations=50):
    logger.info(f"Starting Monte Carlo Analysis ({n_simulations} runs)...")
    
    # 1. Load Data Once
    parquet_file = "data/raw/EURUSD_1m_2023-11-23_2025-11-22.parquet"
    if not os.path.exists(parquet_file):
        logger.error("Data file not found!")
        return
        
    logger.info(f"Loading data from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    # Feature Engineering
    logger.info("Calculating features...")
    fe = FeatureEngineer()
    df = fe.calculate_features(df)
    df = df.dropna().reset_index(drop=False)
    
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Sample Time: {df.iloc[0].get('time', 'N/A')} / {df.iloc[0].get('index', 'N/A')}")
    
    # 2. Run Simulations in Parallel
    logger.info("Running simulations...")
    agent_path = "models/ppo_ftmo_agent.zip"
    
    # Use fewer jobs to avoid memory issues if loading model many times
    results = Parallel(n_jobs=4, verbose=5)(
        delayed(run_simulation)(i, df, agent_path) for i in range(n_simulations)
    )
    
    # 3. Analyze Results
    df_res = pd.DataFrame(results)
    
    pass_rate = len(df_res[df_res['result'] == 'PASS']) / n_simulations * 100
    fail_rate = len(df_res[df_res['result'].str.contains('FAIL')]) / n_simulations * 100
    timeout_rate = len(df_res[df_res['result'] == 'TIMEOUT']) / n_simulations * 100
    
    avg_profit = df_res['profit'].mean() * 100
    avg_dd = df_res['drawdown'].mean() * 100
    max_dd = df_res['drawdown'].max() * 100
    
    logger.info("\n" + "="*40)
    logger.info("MONTE CARLO RESULTS")
    logger.info("="*40)
    logger.info(f"Simulations: {n_simulations}")
    logger.info(f"Pass Rate:   {pass_rate:.1f}%")
    logger.info(f"Fail Rate:   {fail_rate:.1f}%")
    logger.info(f"Timeout:     {timeout_rate:.1f}% (Profitable but <10%)")
    logger.info("-" * 20)
    logger.info(f"Avg Profit:  {avg_profit:.2f}%")
    logger.info(f"Avg Drawdown:{avg_dd:.2f}%")
    logger.info(f"Max Drawdown:{max_dd:.2f}%")
    logger.info("="*40)
    
    # Save detailed results
    df_res.to_csv("data/monte_carlo_results.csv")
    logger.info("Detailed results saved to data/monte_carlo_results.csv")

if __name__ == "__main__":
    run_monte_carlo(n_simulations=20) # Start with 20 for speed

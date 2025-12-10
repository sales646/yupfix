import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from env.ftmo_env import FTMOEvaluator
from src.data.dukascopy_loader import DukascopyLoader
from src.data.feature_engineer import FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PPOBacktest")

def run_backtest(start_date, end_date):
    logger.info(f"--- PPO-Only Backtest ({start_date} to {end_date}) ---")
    
    # 1. Load Data
    loader = DukascopyLoader()
    df = loader.fetch_data("EURUSD", start_date, end_date, resample="1min")
    if df is None: 
        return
    
    # 2. Features
    fe = FeatureEngineer()
    df = fe.calculate_features(df)
    df = df.dropna().reset_index(drop=False)
    
    # Add dummy ML signal if not present
    if 'ml_signal' not in df.columns:
        df['ml_signal'] = 0.5
    
    # 3. Load PPO Agent
    try:
        agent_ppo = PPO.load("models/ppo_ftmo_agent")
        logger.info("✅ Loaded PPO agent with new features (12-dim obs space)")
    except:
        logger.error("❌ PPO Agent not found! Train it first.")
        return

    # 4. Simulation Loop (PPO ONLY - no ensemble)
    env = FTMOEvaluator(df, symbol="EURUSD")
    
    obs, _ = env.reset()
    done = False
    
    logger.info("Running PPO-only simulation...")
    
    while not done:
        # Get PPO Action directly
        action_ppo, _ = agent_ppo.predict(obs, deterministic=True)
        if isinstance(action_ppo, np.ndarray):
            action_ppo = int(action_ppo.item())
        
        # Use PPO action directly (no ensemble)
        final_action = action_ppo
        
        # Log actions occasionally
        if env.current_step % 1000 == 0:
            logger.info(f"Step {env.current_step}: PPO={action_ppo}")
        
        # Step Env
        obs, reward, terminated, truncated, info = env.step(final_action)
        done = terminated or truncated
        
    # 5. Report
    logger.info("--- Backtest Result ---")
    logger.info(f"Final Equity: ${env.equity:.2f}")
    profit_pct = (env.equity - env.initial_balance) / env.initial_balance * 100
    logger.info(f"Profit: ${env.equity - env.initial_balance:.2f} ({profit_pct:.2f}%)")
    logger.info(f"Trades: {env.trades_count}")
    
    if env.equity >= env.initial_balance * 1.10:
        logger.info("RESULT: ✅ PASSED (+10%) 🏆")
        return True
    elif env.equity <= env.initial_balance * 0.90:
        logger.info("RESULT: ❌ FAILED (Max Loss)")
        return False
    else:
        logger.info("RESULT: ⚠️ INCOMPLETE (Time Limit)")
        return None

if __name__ == "__main__":
    # Test on early 2023 (Short range for quick check)
    result = run_backtest("2023-01-01", "2023-01-15")
    
    if result:
        logger.info("\n🎉 Model passes FTMO challenge criteria!")
    elif result is False:
        logger.info("\n⚠️ Model needs more training or tuning")
    else:
        logger.info("\n📊 Model needs longer test period")

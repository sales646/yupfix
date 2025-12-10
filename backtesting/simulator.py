import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from env.ftmo_env import FTMOEvaluator
from models.xgb_model import XGBoostModel
from ensemble.decision_layer import DecisionLayer
from src.data.dukascopy_loader import DukascopyLoader
from src.data.feature_engineer import FeatureEngineer
from models.bc_agent import BCAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Simulator")

def run_backtest(start_date, end_date):
    logger.info(f"--- Starting Backtest ({start_date} to {end_date}) ---")
    
    # 1. Load Data
    loader = DukascopyLoader()
    df = loader.fetch_data("EURUSD", start_date, end_date, resample="1min")
    if df is None: return
    
    # 2. Features & ML
    fe = FeatureEngineer()
    df = fe.calculate_features(df)
    
    model_xgb = XGBoostModel()
    try:
        model_xgb.load()
        xgb_features = model_xgb.prepare_features(df)
        import xgboost as xgb
        dtest = xgb.DMatrix(xgb_features)
        probs = model_xgb.model.predict(dtest)
        df['ml_signal'] = probs[:, 2]
    except:
        df['ml_signal'] = 0.5
        
    df = df.dropna().reset_index(drop=False)
    
    # 3. Load Agents
    try:
        agent_ppo = PPO.load("models/ppo_ftmo_agent")
    except:
        logger.error("PPO Agent not found! Train it first.")
        return

    # Load BC Agent
    bc_agent = BCAgent(input_dim=8, output_dim=3)
    bc_agent.load("models/bc_agent.pth")

    # 4. Simulation Loop
    env = FTMOEvaluator(df)
    decision_layer = DecisionLayer()
    
    obs, _ = env.reset()
    done = False
    
    logger.info("Running Simulation...")
    
    while not done:
        # Get PPO Action
        action_ppo, _ = agent_ppo.predict(obs, deterministic=True)
        if isinstance(action_ppo, np.ndarray):
            action_ppo = int(action_ppo.item())
        
        # Get ML Signal
        # Extract from obs (last element is ml_signal)
        ml_prob = obs[-1]
        
        # Get BC Action
        # obs is numpy array, need to reshape? BCAgent handles it.
        # Note: obs includes ML signal at end. BCAgent trained on 8 features.
        action_bc = bc_agent.predict(obs)
        
        # Ensemble Decision
        final_action, conf = decision_layer.get_decision(obs, action_ppo, action_bc, ml_prob)
        
        # Log actions occasionally
        if env.current_step % 1000 == 0:
            logger.info(f"Step {env.current_step}: PPO={action_ppo}, BC={action_bc}, ML={ml_prob:.2f} -> Final={final_action}")
        
        # Step Env
        # Note: Env expects 0,1,2. DecisionLayer returns 0,1,2.
        obs, reward, terminated, truncated, info = env.step(final_action)
        done = terminated or truncated
        
    # 5. Report
    logger.info("--- Backtest Result ---")
    logger.info(f"Final Equity: {env.equity:.2f}")
    logger.info(f"Profit: {(env.equity - env.initial_balance):.2f} ({(env.equity - env.initial_balance)/env.initial_balance*100:.2f}%)")
    logger.info(f"Trades: {env.trades_count}")
    
    if env.equity >= env.initial_balance * 1.10:
        logger.info("RESULT: PASSED (+10%) 🏆")
    elif env.equity <= env.initial_balance * 0.90:
        logger.info("RESULT: FAILED (Max Loss) ❌")
    else:
        logger.info("RESULT: INCOMPLETE (Time Limit) ⚠️")

if __name__ == "__main__":
    # Test on early 2023 (Short range for quick check)
    run_backtest("2023-01-01", "2023-01-15")

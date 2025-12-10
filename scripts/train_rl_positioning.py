"""
Train RL Agent for Position Sizing
Uses PPO to optimize position sizing policy on top of Mamba features.
"""
import sys
import os
from pathlib import Path
import yaml
import torch
import pandas as pd
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.mamba_actor_critic import MambaActorCritic
from src.rl.ppo_trainer import PPOTrainer
from env.ftmo_env import FTMOEvaluator
from src.yup250_pipeline import Yup250Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RLTraining")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class MambaPortfolioEnv(FTMOEvaluator):
    """
    Wrapper for FTMOEvaluator to support continuous actions for PPO.
    Action: [-1, 1] -> Position Size (Short to Long)
    """
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        # Override action space for continuous actions
        # We use a single continuous action for position size/direction
        from gymnasium import spaces
        import numpy as np
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
    def step(self, action):
        # Action is continuous [-1, 1]
        # Map to direction and size
        # > 0: Long, < 0: Short
        # Magnitude: Size (0 to 1)
        
        act_val = action[0]
        direction = 0
        if act_val > 0.1:
            direction = 1 # Buy
        elif act_val < -0.1:
            direction = 2 # Sell
        else:
            direction = 0 # Hold
            
        # We need to hack the step method or override it completely to use the magnitude
        # For now, let's use the discrete step but we lose the sizing granularity
        # Ideally, we should modify FTMOEvaluator to accept size
        
        # HACK: Store the magnitude for the environment to use
        self.current_action_magnitude = abs(act_val)
        
        # Call parent step
        return super().step(direction)
    
    def calculate_position_size(self, atr_value, confidence):
        # Override to use the RL agent's magnitude
        # Base max lots on equity
        max_leverage_lots = (self.equity * 4) / self.contract_size
        
        # Scale by action magnitude
        lots = max_leverage_lots * self.current_action_magnitude
        
        # Still apply risk checks if needed, but RL should learn this
        return max(0.01, round(lots, 2))

def main():
    config = load_config("config/training.yaml")
    
    # Load Data using Pipeline
    pipeline = Yup250Pipeline(config)
    data = pipeline.load_data(config['data']['train_path'])
    features = pipeline.prepare_features(data)
    
    # Use EURUSD for now
    symbol = "EURUSD"
    if symbol not in features:
        logger.error(f"{symbol} not found in data")
        return
        
    df = features[symbol]
    
    # Initialize Environment
    env = MambaPortfolioEnv(df, symbol=symbol)
    
    # Initialize Policy
    mamba_config = config['model']
    policy = MambaActorCritic(mamba_config, action_dim=1)
    
    # Initialize Trainer
    ppo_config = {
        'lr': 3e-4,
        'gamma': 0.99,
        'clip_ratio': 0.2,
        'ppo_epochs': 10,
        'batch_size': 64
    }
    
    trainer = PPOTrainer(env, policy, ppo_config)
    
    # Train
    logger.info("Starting RL Training...")
    trainer.train(total_timesteps=100000)
    
    # Save
    torch.save(policy.state_dict(), "models/rl_policy.pt")
    logger.info("Saved policy to models/rl_policy.pt")

if __name__ == "__main__":
    main()

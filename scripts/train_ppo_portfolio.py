"""
Train PPO Portfolio Agent
Main script to train the RL agent for portfolio management.
"""
import argparse
import yaml
import torch
import pandas as pd
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yup250_pipeline import Yup250Pipeline
from src.rl.mamba_portfolio_env import MambaPortfolioEnv
from src.rl.ppo_agent import PPOAgent
from src.contracts.config import RiskConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    if 'risk' in raw:
        RiskConfig(**raw['risk'])
    return raw

def main(args):
    config = load_config(args.config)
    
    # 1. Load Data & Model
    logger.info("Loading Pipeline and Data...")
    pipeline = Yup250Pipeline(config)
    
    # Load Train Data
    # For RL, we might want a specific period or use the same train split
    train_data = pipeline.load_data(config['data']['train_path'])
    
    # Load Pre-trained Mamba Model
    logger.info("Loading Pre-trained Mamba Model...")
    # Ideally, load from checkpoint
    # model = pipeline.create_model(...)
    # checkpoint = torch.load(args.model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval() # Freeze Mamba
    
    # MOCK for now since we don't have a trained model file in this session
    class MockMamba(torch.nn.Module):
        def forward(self, x):
            # Return dummy dict
            return {'uncertainty': torch.rand(x.shape[0])} # Mock
    model = MockMamba()
    
    # 2. Setup Environment
    logger.info("Setting up RL Environment...")
    # We need to pass data as a dict of DataFrames {symbol: df}
    # Yup250Pipeline loads data as a single DF or dict? 
    # load_data returns a dict {symbol: df} usually.
    
    env = MambaPortfolioEnv(model, train_data, config)
    
    # 3. Setup Agent
    logger.info("Initializing PPO Agent...")
    agent = PPOAgent(env, config)
    
    # 4. Train
    logger.info(f"Starting Training for {args.timesteps} timesteps...")
    agent.train(total_timesteps=args.timesteps)
    
    # 5. Save Agent
    save_path = Path(config['logging']['save_dir']) / "ppo_agent.pt"
    torch.save(agent.actor.state_dict(), save_path)
    logger.info(f"Agent saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/training.yaml')
    parser.add_argument('--model-path', type=str, required=False, help='Path to pretrained Mamba model')
    parser.add_argument('--timesteps', type=int, default=100000)
    args = parser.parse_args()
    main(args)

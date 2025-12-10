import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from models.bc_agent import BCAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BCTrainer")

def train_bc():
    logger.info("Loading expert data...")
    try:
        obs = np.load("data/expert_obs.npy")
        actions = np.load("data/expert_actions.npy")
    except FileNotFoundError:
        logger.error("Expert data not found! Run scripts/generate_expert_data.py first.")
        return

    logger.info(f"Loaded {len(obs)} samples.")
    
    # Initialize Agent
    # Input dim = 12 (matches environment observation space)
    # Output dim = 3 (Hold, Buy, Sell)
    agent = BCAgent(input_dim=12, output_dim=3, lr=0.001)
    
    # Train
    logger.info("Starting BC Training...")
    agent.train(obs, actions, epochs=50, batch_size=64)
    
    # Save
    agent.save("models/bc_agent.pth")
    logger.info("BC Agent saved to models/bc_agent.pth")

if __name__ == "__main__":
    train_bc()

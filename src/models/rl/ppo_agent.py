from stable_baselines3 import PPO
from src.models.rl.environment import FTMOEvaluator
import pandas as pd
import os

class PPOAgent:
    def __init__(self, model_path: str = "models/ppo_agent"):
        self.model_path = model_path
            self.load()
        
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def save(self):
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        self.model.save(self.model_path)

    def load(self):
        if os.path.exists(self.model_path + ".zip"):
            self.model = PPO.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

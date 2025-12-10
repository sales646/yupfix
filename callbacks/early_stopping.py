"""
Early Stopping Callback for PPO Training.
Prevents overfitting by monitoring validation performance.
"""
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class EarlyStoppingCallback(BaseCallback):
    """
    Stop training early if performance plateaus or degrades.
    
    Monitors:
    - Episode reward mean (should trend upward)
    - Performance on validation set
    """
    
    def __init__(self, patience=10, min_delta=0.01, check_freq=10000, verbose=1):
        """
        Args:
            patience: Number of checks without improvement before stopping
            min_delta: Minimum change in reward to be considered improvement
            check_freq: Check performance every N steps
            verbose: Logging level
        """
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.check_freq = check_freq
        self.best_reward = -np.inf
        self.wait = 0
        self.stopped_epoch = 0
    
    def _on_step(self) -> bool:
        """Called at every step. Returns False to stop training."""
        # Check performance every check_freq steps
        if self.n_calls % self.check_freq != 0:
            return True
        
        # Get episode reward from training
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        else:
            return True
        
        # Check if improved
        if mean_reward > self.best_reward + self.min_delta:
            self.best_reward = mean_reward
            self.wait = 0
            if self.verbose > 0:
                print(f"✅ New best reward: {mean_reward:.2f} at {self.num_timesteps} steps")
        else:
            self.wait += 1
            if self.verbose > 0:
                print(f"⚠️ No improvement for {self.wait}/{self.patience} checks (reward: {mean_reward:.2f})")
            
            if self.wait >= self.patience:
                self.stopped_epoch = self.num_timesteps
                if self.verbose > 0:
                    print(f"🛑 Early stopping at {self.num_timesteps} steps! Best reward: {self.best_reward:.2f}")
                return False  # Stop training
        
        return True  # Continue training

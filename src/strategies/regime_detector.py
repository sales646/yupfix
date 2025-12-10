import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib
import os
from typing import Dict

class RegimeDetector:
    def __init__(self, model_path: str = "models/hmm_regime.pkl"):
        self.model_path = model_path
        self.model = None
        self.n_components = 3 # 0: Low Vol/Range, 1: Trend, 2: High Vol/Crash
        
    def train(self, df: pd.DataFrame):
        """Train HMM on returns and volatility."""
        # Feature Engineering
        returns = df['close'].pct_change().dropna()
        vol = returns.rolling(20).std().dropna()
        
        # Align
        common_idx = returns.index.intersection(vol.index)
        X = pd.concat([returns.loc[common_idx], vol.loc[common_idx]], axis=1).values
        
        # Train HMM
        self.model = GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=100)
        self.model.fit(X)
        
        self.save()
        
    def predict(self, df: pd.DataFrame) -> int:
        """Predict current regime (0, 1, or 2)."""
        if self.model is None:
            self.load()
            
        returns = df['close'].pct_change().dropna()
        vol = returns.rolling(20).std().dropna()
        
        if len(returns) < 20: return 0 # Default
        
        common_idx = returns.index.intersection(vol.index)
        X = pd.concat([returns.loc[common_idx], vol.loc[common_idx]], axis=1).values
        
        hidden_states = self.model.predict(X)
        return hidden_states[-1]

    def save(self):
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        joblib.dump(self.model, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            # Initialize untrained if not found (for dev)
            self.model = GaussianHMM(n_components=self.n_components, covariance_type="full")
            # raise FileNotFoundError(f"Model not found at {self.model_path}")

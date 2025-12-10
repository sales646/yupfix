"""
Mamba Portfolio Environment for PPO Training
Integrates Mamba model predictions, uncertainty scaling, and Risk Manager.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from typing import Dict, Any, List
import logging

from ..risk.manager import RiskManager

logger = logging.getLogger("MambaPortfolioEnv")

class MambaPortfolioEnv(gym.Env):
    """
    RL Environment for Portfolio Management.
    
    State: Mamba hidden state + Account state
    Action: Portfolio weights [-1, 1] per symbol
    Reward: Risk-adjusted return (Sharpe/Sortino)
    """
    
    def __init__(self, mamba_model, data: Dict[str, pd.DataFrame], config: Dict[str, Any]):
        super().__init__()
        
        self.mamba = mamba_model
        self.data = data
        self.config = config
        self.symbols = list(data.keys())
        self.n_assets = len(self.symbols)
        
        # Initialize Risk Manager
        self.guardian = RiskManager(config.get('risk', {}))
        
        # Action Space: Continuous weights [-1, 1] for each asset
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        
        # Observation Space: 
        # We need to define what the agent sees.
        # Ideally: Mamba features (d_model) + Account State
        # For simplicity, let's assume Mamba features are d_model per asset
        self.d_model = config['model']['d_model']
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_assets * self.d_model + 5,), dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.max_steps = min(len(df) for df in data.values()) - 1
        self.initial_balance = 100000.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.positions = {sym: 0.0 for sym in self.symbols}
        self.prev_actions = np.zeros(self.n_assets)
        
        # Costs
        self.transaction_cost_bps = config.get('costs', {}).get('transaction_bps', 5.0) # 5 bps default
        self.holding_cost_bps = config.get('costs', {}).get('holding_bps', 0.0)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 100 # Warmup
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.positions = {sym: 0.0 for sym in self.symbols}
        self.prev_actions = np.zeros(self.n_assets)
        
        return self._get_observation(), {}
        
    def _get_observation(self):
        # Get Mamba features for current step
        # We need to construct the input tensor for Mamba
        # This is complex because Mamba expects a sequence.
        # We assume 'data' contains pre-processed features ready for Mamba.
        
        # Placeholder for feature extraction
        # In a real implementation, we would pass the recent history window to Mamba
        # and get the last hidden state.
        
        # For this mock implementation, we return random features
        features = np.zeros(self.n_assets * self.d_model, dtype=np.float32)
        
        # Account state
        account_state = np.array([
            self.balance / self.initial_balance,
            self.equity / self.initial_balance,
            (self.peak_equity - self.equity) / self.peak_equity, # Drawdown
            0.0, # Daily PnL (placeholder)
            0.0  # Exposure (placeholder)
        ], dtype=np.float32)
        
        return np.concatenate([features, account_state])
        
    def step(self, action):
        # action: Portfolio weights [-1, 1] per symbol
        
        # 1. Get Mamba Predictions & Uncertainty
        # In a real scenario, we run the model forward here.
        # For now, we simulate the uncertainty scaling logic.
        
        # Mock uncertainty (0 to 1)
        uncertainty = np.random.random(self.n_assets) 
        
        # Scale positions by inverse uncertainty
        # High uncertainty -> Low confidence scale
        confidence_scale = 1.0 / (1.0 + uncertainty * 5.0) # Scale factor
        scaled_action = action * confidence_scale
        
        # 2. Apply Guardian Limits
        final_positions = {}
        for i, sym in enumerate(self.symbols):
            target = scaled_action[i]
            
            # Check individual limits
            allowed = self.guardian.check_position(
                target, self.balance, self.peak_equity, self.initial_balance
            )
            final_positions[sym] = allowed * np.sign(target)
            
        # Check correlation limits
        final_positions = self.guardian.check_correlation_exposure(final_positions)
        
        # 3. Simulate Market Step
        # Calculate PnL based on position changes and price moves
        step_pnl = 0.0
        transaction_costs = 0.0
        
        # Calculate turnover (change in weights)
        # We approximate turnover cost based on notional value changed
        # Delta Weight * Equity * Cost
        action_diff = np.abs(scaled_action - self.prev_actions)
        turnover_cost = np.sum(action_diff) * self.equity * (self.transaction_cost_bps / 10000.0)
        transaction_costs += turnover_cost
        
        self.prev_actions = scaled_action
        
        for i, sym in enumerate(self.symbols):
            df = self.data[sym]
            current_price = df.iloc[self.current_step]['close']
            next_price = df.iloc[self.current_step + 1]['close']
            ret = (next_price - current_price) / current_price
            
            pos = final_positions[sym]
            # PnL = Position * Return * Capital
            # Assuming Position 1.0 = 100% Equity
            step_pnl += pos * ret * self.equity
            
        # Deduct costs
        step_pnl -= transaction_costs
            
        self.equity += step_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.current_step += 1
        
        # 4. Compute Reward
        # Reward = Log Return - Volatility Penalty - Turnover Penalty
        # We already deducted costs from PnL, so Log Return reflects it.
        # But we can add an extra penalty for "useless" turnover if needed.
        # For now, let's stick to PnL-based reward which implicitly penalizes costs.
        
        if self.equity <= 0:
             reward = -100
             terminated = True
        else:
             reward = np.log(self.equity / (self.equity - step_pnl)) * 100
        
        # Terminate if broke (50% drawdown)
        terminated = False
        if self.equity < self.initial_balance * 0.5:
            terminated = True
            reward = -100
            
        truncated = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, terminated, truncated, {}

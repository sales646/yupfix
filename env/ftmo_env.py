import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Optional
import logging

# Setup logger
logger = logging.getLogger("FTMOEnv")

from src.execution.execution_model import ExecutionModel

class FTMOEvaluator(gym.Env):
    """
    Custom Environment that follows gym interface.
    Simulates FTMO Challenge rules:
    - Max Daily Loss: 5%
    - Max Total Loss: 10%
    - Profit Target: 10%
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 100000.0, 
                 profit_target: float = 0.10, max_loss_limit: float = 0.10, symbol: str = "EURUSD"):
        """
        Initialize FTMO Environment.
        
        Args:
            df: DataFrame with OHLCV and features
            initial_balance: Starting balance ($100k default)
            profit_target: Target profit % (default 0.10 = 10%)
            max_loss_limit: Max total loss % (default 0.10 = 10%)
            symbol: Trading symbol for ATR-based position sizing
        """
        super(FTMOEvaluator, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.symbol = symbol
        
        # Action Space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation Space (EXPANDED): 
        # [ret_1, ret_5, garch_vol, rsi, macd, dist_ma, dd_pct, ml_signal, atr, hour_sin, hour_cos, session]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        
        # State
        self.current_step = 0
        self.balance = initial_balance
        self.equity = initial_balance
        self.start_of_day_equity = initial_balance
        self.position = 0 # 0=Flat, 1=Long, -1=Short
        self.entry_price = 0.0
        self.current_lots = 0.0 # Track position size
        self.trades_count = 0
        self.max_steps = len(df) - 1
        
        # Contract Size
        if self.symbol == "EURUSD":
            self.contract_size = 100000.0
        elif self.symbol == "XAUUSD":
            self.contract_size = 100.0
        else:
            self.contract_size = 100000.0
        
        # FTMO Limits (NOW CONFIGURABLE for Curriculum Learning)
        self.daily_loss_limit = 0.05  # Always 5% (FTMO rule)
        self.max_loss_limit = max_loss_limit  # Configurable
        self.profit_target = profit_target  # Configurable
        
        # Execution Model
        self.execution_model = ExecutionModel(avg_slippage_pips=0.5, fill_rate=0.98)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 20 # Start after warm-up
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.start_of_day_equity = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.current_lots = 0.0
        self.trades_count = 0
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Get data for current step
        row = self.df.iloc[self.current_step]
        
        features = [
            row.get('ret_1', 0),
            row.get('ret_5', 0),
            row.get('garch_vol', 0),
            row.get('rsi', 50),
            row.get('macd', 0),
            row.get('dist_ma_20', 0),
            # Account State
            (self.initial_balance - self.equity) / self.initial_balance, # Drawdown %
            row.get('ml_signal', 0), # ML Forecast
            # NEW: ATR and Time Features
            row.get('atr', 0),
            row.get('hour_sin', 0),
            row.get('hour_cos', 0),
            row.get('session', 0)
      ]
        
        return np.array(features, dtype=np.float32)
    
    def calculate_position_size(self, atr_value: float, confidence: float = 0.0) -> float:
        """
        ATR-based adaptive position sizing with Confidence Scaling.
        
        Args:
            atr_value: Current ATR value
            confidence: Confidence score (0.0 to 1.0) derived from ML signal
            
        Returns:
            Lot size (capped at 1:4 leverage)
        """
        # Aggressive Sizing:
        # Base risk 0.5% -> Scales up to 2% based on confidence
        base_risk_pct = 0.005 
        risk_multiplier = 1.0 + (confidence * 3.0) # 1x to 4x
        target_risk_pct = base_risk_pct * risk_multiplier
        
        # Symbol-specific ATR normalization
        if self.symbol == "EURUSD":
            atr_threshold = 0.0005  # 5 pips baseline
            pip_value = 10  # $10 per pip for 1 standard lot
            lot_multiplier = 1.0
        elif self.symbol == "XAUUSD":
            atr_threshold = 0.5  # $0.50 baseline
            pip_value = 1  # $1 per pip for gold
            lot_multiplier = 0.1  # Gold needs smaller lots
        else:
            atr_threshold = 0.0005
            pip_value = 10
            lot_multiplier = 1.0
        
        # Volatility factor: reduce size as ATR increases
        if atr_value > 0:
            volatility_factor = min(atr_threshold / atr_value, 2.0)
        else:
            volatility_factor = 1.0
        
        # Calculate risk amount
        risk_amount = self.equity * target_risk_pct
        
        # Stop loss = 2x ATR
        stop_loss = max(atr_value * 2, atr_threshold)
        
        # Calculate lots
        lots = (risk_amount / (stop_loss * pip_value)) * lot_multiplier * volatility_factor
        
        # --- LEVERAGE CAP (1:4) ---
        # Max Exposure = Equity * 4
        # Max Lots = (Equity * 4) / Contract Size
        max_leverage_lots = (self.equity * 4) / self.contract_size
        
        # Apply Cap
        lots = min(lots, max_leverage_lots)
        
        return max(0.01, round(lots, 2))

    def step(self, action):
        self.current_step += 1
        
        # --- Daily Loss Reset Logic ---
        # Check for new day to reset start_of_day_equity
        current_date = None
        if 'time' in self.df.columns:
            current_date = pd.to_datetime(self.df.iloc[self.current_step]['time']).date()
        elif 'timestamp' in self.df.columns:
            current_date = pd.to_datetime(self.df.iloc[self.current_step]['timestamp']).date()
        elif 'index' in self.df.columns:
            current_date = pd.to_datetime(self.df.iloc[self.current_step]['index']).date()
            
        if current_date is not None:
            prev_date = None
            if self.current_step > 0:
                if 'time' in self.df.columns:
                    prev_date = pd.to_datetime(self.df.iloc[self.current_step - 1]['time']).date()
                elif 'timestamp' in self.df.columns:
                    prev_date = pd.to_datetime(self.df.iloc[self.current_step - 1]['timestamp']).date()
                elif 'index' in self.df.columns:
                    prev_date = pd.to_datetime(self.df.iloc[self.current_step - 1]['index']).date()
            
            if prev_date is not None and current_date != prev_date:
                self.start_of_day_equity = self.equity

        # 1. Market Step
        row = self.df.iloc[self.current_step]
        current_price = row['close']
        prev_price = self.df.iloc[self.current_step - 1]['close']
        
        # Extract Confidence from ML Signal (0.5 = neutral)
        ml_signal = row.get('ml_signal', 0.5)
        confidence = abs(ml_signal - 0.5) * 2.0 # 0.0 to 1.0
        
        # 2. PnL Calculation (Based on Lots)
        price_change = current_price - prev_price
        
        prev_equity = self.equity
        
        if self.position == 1: # Long
            pnl = self.current_lots * self.contract_size * price_change
        elif self.position == -1: # Short
            pnl = self.current_lots * self.contract_size * -price_change
        else:
            pnl = 0
            
        self.equity += pnl
        
        # 3. Execute Action (for next step)
        prev_position = self.position
        
        # Get execution parameters
        spread = row.get('spread_avg', 0.0001) # Default to 1 pip if missing
        
        if action == 1: # Buy
            if self.position != 1: # Entry or Flip
                atr = row.get('atr', 0)
                self.current_lots = self.calculate_position_size(atr, confidence)
                
                # EXECUTION SIMULATION
                exec_price = self.execution_model.execute(1, current_price, spread)
                if exec_price is not None:
                    self.entry_price = exec_price
                    self.position = 1
                else:
                    # Order rejected
                    self.position = 0 # Stay flat or keep previous? 
                    # If flip rejected, we might be stuck in previous position or flat.
                    # For simplicity, if rejected, we don't enter.
                    pass
            else:
                 self.position = 1 # Already long
            
        elif action == 2: # Sell
            if self.position != -1: # Entry or Flip
                atr = row.get('atr', 0)
                self.current_lots = self.calculate_position_size(atr, confidence)
                
                # EXECUTION SIMULATION
                exec_price = self.execution_model.execute(-1, current_price, spread)
                if exec_price is not None:
                    self.entry_price = exec_price
                    self.position = -1
                else:
                    pass
            else:
                self.position = -1 # Already short
            
        else: # Hold/Flat
            self.position = 0
            self.current_lots = 0.0
            
        # 4. Reward Calculation
        terminated = False
        truncated = False
        reward = 0
        
        # Track Trades (Entry)
        if self.position != 0 and self.position != prev_position:
            self.trades_count += 1
        
        # --- FTMO RULES CHECK ---
        
        # Max Total Loss
        if self.equity < self.initial_balance * (1 - self.max_loss_limit):
            self.position = 0  # Close all positions immediately
            reward = -100
            terminated = True
            logger.info(f"❌ Max loss hit! Equity: ${self.equity:.2f}")
            
        # Max Daily Loss (5%)
        daily_drawdown = (self.start_of_day_equity - self.equity) / self.start_of_day_equity
        if daily_drawdown >= self.daily_loss_limit:
            self.position = 0  # Close all positions
            reward = -100
            terminated = True
            logger.info(f"❌ Daily loss limit hit! Drawdown: {daily_drawdown*100:.2f}%")
            
        # Target Reached
        elif self.equity >= self.initial_balance * (1 + self.profit_target):
            self.position = 0  # Close all positions on target
            reward = 100 * (self.profit_target / 0.10)
            terminated = True 
            logger.info(f"✅ Target reached! Equity: ${self.equity:.2f} (+{self.profit_target*100:.0f}%)")
            
        else:
            # Standard Reward: PnL Change (scaled)
            reward = (self.equity - prev_equity) / self.initial_balance * 100
            
            # STRONG Inactivity Penalty (Cost of Living)
            if self.position == 0:
                reward -= 0.1 
            
            # Exploration Bonus
            if action != 0 and prev_position == 0:
                reward += 0.05 
            
        if self.current_step >= self.max_steps:
            truncated = True
            # Harsh penalty for not reaching goal
            if self.trades_count < 5:
                reward -= 50
            if self.equity < self.initial_balance * 1.10:
                reward -= 20 
            
        return self._next_observation(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Equity: {self.equity:.2f}, Pos: {self.position}")

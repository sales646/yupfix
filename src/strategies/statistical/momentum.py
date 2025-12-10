import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.strategies.statistical.garch import GarchModel

class MomentumStrategy:
    def __init__(self, lookback: int = 20, vol_target: float = 0.15):
        self.lookback = lookback
        self.vol_target = vol_target # Annualized Vol Target (e.g., 15%)
        self.garch = GarchModel()

    def calculate_signal(self, candles: pd.DataFrame) -> Dict:
        """
        Calculate momentum signal and position size.
        Returns: {'action': 'BUY'/'SELL'/'HOLD', 'volume': float}
        """
        if len(candles) < self.lookback + 100: # Need history for GARCH
            return {'action': 'HOLD', 'volume': 0.0}
            
        # Calculate Returns
        closes = candles['close']
        returns = closes.pct_change().dropna()
        
        # 1. Fit GARCH
        self.garch.fit(returns)
        vol_forecast = self.garch.forecast()
        
        # Avoid division by zero
        if vol_forecast < 0.001:
            vol_forecast = 0.001
            
        # 2. Calculate Momentum (Simple Returns over lookback)
        momentum = closes.iloc[-1] / closes.iloc[-1 - self.lookback] - 1
        
        # 3. Determine Direction
        action = 'HOLD'
        if momentum > 0:
            action = 'BUY'
        elif momentum < 0:
            action = 'SELL'
            
        # 4. Volatility Scaling (Target Vol / Realized Vol)
        # Position = (Target Vol / Forecast Vol) * Capital / Price
        # Here we return a "leverage factor" or raw volume if we knew capital
        # For now, return a scaling factor
        
        scaling_factor = self.vol_target / (vol_forecast * np.sqrt(252 * 1440)) # Annualize 1-min vol? 
        # Note: GARCH forecast is usually per-step. If step is 1-min, need to scale carefully.
        # Let's assume GARCH forecast is for the *next minute*.
        # Annualized Vol = Minute Vol * sqrt(252 * 1440)
        # Wait, GarchModel.forecast returns "decimal". 
        # If fitted on 1-min returns, it returns 1-min vol.
        
        # Let's simplify: Use realized vol for scaling to be robust
        # But user asked for GARCH.
        
        # Correct Annualization:
        # vol_annual = vol_minute * sqrt(252 * 24 * 60)
        vol_annual = vol_forecast * np.sqrt(252 * 1440)
        
        leverage = self.vol_target / vol_annual
        
        # Cap leverage to FTMO limits (e.g., max 30x, but practically 5-10x)
        leverage = min(leverage, 5.0)
        
        return {
            'action': action,
            'leverage': leverage,
            'momentum_score': momentum,
            'vol_forecast_annual': vol_annual
        }

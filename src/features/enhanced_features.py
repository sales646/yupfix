"""
Enhanced features including Delay Zone Targets
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_delay_zone_targets(df: pd.DataFrame, horizon: int = 12, delay: int = 3) -> pd.Series:
    """
    Create Delay Zone Targets.
    Ignores immediate noise (t+1 to t+delay) and targets direction at t+horizon.
    
    Formula: target = sign(close[t+horizon] - close[t+delay])
    
    Args:
        df: DataFrame with 'close' column
        horizon: Target horizon (e.g. 12 bars)
        delay: Delay bars to ignore (e.g. 3 bars)
        
    Returns:
        Series with -1, 0, 1 labels
    """
    close = df['close']
    future_close = close.shift(-horizon)
    delayed_close = close.shift(-delay)
    diff = future_close - delayed_close
    targets = np.sign(diff)
    targets = targets.fillna(0).astype(int)
    return targets

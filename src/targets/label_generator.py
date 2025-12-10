"""
Label Generator for YUP-250 Trading System
Creates multi-task labels from OHLCV data
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def create_trading_labels(df: pd.DataFrame, 
                          horizon: int = 12, 
                          delay: int = 3) -> pd.DataFrame:
    """
    Create multi-task labels from OHLCV data.
    
    Labels:
    - direction: Delay Zone target (-1/0/+1 mapped to 0/1/2)
    - volatility: Quantile of rolling std (0/1/2)
    - magnitude: Normalized absolute return [0,1]
    
    Args:
        df: DataFrame with 'close' column
        horizon: Target horizon in bars (default 12 = 12 seconds)
        delay: Delay bars to ignore noise (default 3 = 3 seconds)
        
    Returns:
        DataFrame with columns [direction, volatility, magnitude]
    """
    labels = pd.DataFrame(index=df.index)
    
    # 1. Direction (Delay Zone)
    # target = sign(close[t+horizon] - close[t+delay])
    
    # Handle multi-timeframe column names
    close_col = 'close'
    if 'close' not in df.columns and 'close_1min' in df.columns:
        close_col = 'close_1min'
        
    if close_col not in df.columns:
        raise KeyError(f"Column '{close_col}' not found in DataFrame")
        
    future_close = df[close_col].shift(-horizon)
    delayed_close = df[close_col].shift(-delay)
    direction_raw = np.sign(future_close - delayed_close)
    
    # Map: -1 -> 0 (DOWN), 0 -> 1 (FLAT), +1 -> 2 (UP)
    labels['direction'] = (direction_raw + 1).fillna(1).astype(int)
    
    # 2. Volatility Regime
    # Rolling std, then quantile bins
    vol_window = 60  # 1 minute
    volatility = df[close_col].pct_change().rolling(vol_window).std()
    
    # Quantile-based binning (0=low, 1=medium, 2=high)
    vol_33 = volatility.quantile(0.33)
    vol_66 = volatility.quantile(0.66)
    
    labels['volatility'] = pd.cut(
        volatility, 
        bins=[-np.inf, vol_33, vol_66, np.inf],
        labels=[0, 1, 2]
    ).astype(float).fillna(1).astype(int)
    
    # 3. Magnitude
    # Normalized absolute return [0, 1]
    returns = (future_close - df[close_col]) / df[close_col]
    abs_returns = returns.abs()
    
    # Clip at 1% and normalize
    max_return = 0.01
    labels['magnitude'] = (abs_returns.clip(0, max_return) / max_return).fillna(0)
    
    # Drop NaN rows (from shift operations)
    # Keep original index for alignment
    labels = labels.dropna()
    
    return labels


def create_labels_for_symbols(data: Dict[str, pd.DataFrame],
                               horizon: int = 12,
                               delay: int = 3) -> Dict[str, pd.DataFrame]:
    """
    Create labels for multiple symbols.
    
    Args:
        data: Dict mapping symbol to OHLCV DataFrame
        horizon: Target horizon in bars
        delay: Delay bars to ignore noise
        
    Returns:
        Dict mapping symbol to labels DataFrame
    """
    labels = {}
    
    for symbol, df in data.items():
        logger.info(f"Creating labels for {symbol}...")
        labels[symbol] = create_trading_labels(df, horizon, delay)
        logger.info(f"{symbol} labels: {labels[symbol].shape}")
        
    return labels


def align_features_and_labels(features: Dict[str, pd.DataFrame],
                               labels: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Align features and labels to have matching indices.
    
    This is necessary because:
    - Labels are created with future shifts (lose last N rows)
    - Features may have NaN from rolling windows (lose first N rows)
    
    Args:
        features: Dict mapping symbol to features DataFrame
        labels: Dict mapping symbol to labels DataFrame
        
    Returns:
        Tuple of (aligned_features, aligned_labels)
    """
    aligned_features = {}
    aligned_labels = {}
    
    for symbol in features.keys():
        if symbol not in labels:
            logger.warning(f"Symbol {symbol} not found in labels, skipping")
            continue
            
        feat = features[symbol]
        lab = labels[symbol]
        
        # Get common index
        common_idx = feat.index.intersection(lab.index)
        
        if len(common_idx) == 0:
            logger.warning(f"No common indices for {symbol}, skipping")
            continue
        
        aligned_features[symbol] = feat.loc[common_idx]
        aligned_labels[symbol] = lab.loc[common_idx]
        
        logger.info(f"{symbol} aligned: {len(common_idx)} samples")
        
    return aligned_features, aligned_labels


def create_multi_horizon_labels(df: pd.DataFrame,
                                 horizons: Dict[str, int] = None,
                                 delay: int = 3) -> Dict[str, pd.DataFrame]:
    """
    Create labels for multiple horizons (optional extension).
    
    Args:
        df: OHLCV DataFrame
        horizons: Dict mapping horizon name to bars (e.g., {'1min': 60, '5min': 300})
        delay: Delay bars
        
    Returns:
        Dict mapping horizon name to labels DataFrame
    """
    if horizons is None:
        horizons = {
            '1min': 60,
            '5min': 300,
            '15min': 900,
            '1hour': 3600
        }
    
    labels_by_horizon = {}
    
    for horizon_name, horizon_bars in horizons.items():
        labels_by_horizon[horizon_name] = create_trading_labels(df, horizon_bars, delay)
        
    return labels_by_horizon

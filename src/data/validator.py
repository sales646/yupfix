"""
Data Validator
Ensures data integrity before feeding it to the model.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("DataValidator")

class DataValidator:
    """
    Checks for common data issues:
    - Missing timestamps (gaps)
    - NaN/Inf values
    - Price anomalies (spikes)
    - Volume anomalies (zero volume)
    """
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.max_gap_seconds = self.config.get('max_gap_seconds', 300) # 5 min gap allowed
        self.max_price_change_pct = self.config.get('max_price_change_pct', 0.05) # 5% move in 1 bar is suspicious
        
    def validate(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """
        Run all checks.
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        if df.empty:
            return False, ["DataFrame is empty"]
            
        # 1. Check Required Columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return False, [f"Missing columns: {missing}"]
            
        # 2. Check NaNs
        if df[required].isnull().any().any():
            nan_counts = df[required].isnull().sum()
            errors.append(f"NaN values detected: {nan_counts[nan_counts > 0].to_dict()}")
            
        # 3. Check Timestamp Gaps
        if isinstance(df.index, pd.DatetimeIndex):
            time_diff = df.index.to_series().diff().dt.total_seconds()
            gaps = time_diff[time_diff > self.max_gap_seconds]
            if not gaps.empty:
                errors.append(f"Found {len(gaps)} gaps larger than {self.max_gap_seconds}s. Max gap: {gaps.max()}s")
        else:
            errors.append("Index is not DatetimeIndex")
            
        # 4. Check Price Anomalies (Spikes)
        pct_change = df['close'].pct_change().abs()
        spikes = pct_change[pct_change > self.max_price_change_pct]
        if not spikes.empty:
            errors.append(f"Found {len(spikes)} price spikes > {self.max_price_change_pct*100}%. Max: {spikes.max()*100:.2f}%")
            
        # 5. Check Zero Volume (if market is open)
        # This is tricky without market hours, but let's warn if > 10% is zero
        zero_vol = (df['volume'] == 0).mean()
        if zero_vol > 0.10:
            errors.append(f"High zero volume count: {zero_vol:.1%}")
            
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Validation FAILED for {symbol}:")
            for e in errors:
                logger.warning(f"  - {e}")
        else:
            logger.info(f"Validation PASSED for {symbol}")
            
        return is_valid, errors

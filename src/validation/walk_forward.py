"""
Walk-Forward Validation Module
Provides temporal cross-validation for time-series data.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Generator

logger = logging.getLogger("WalkForwardValidator")

class WalkForwardValidator:
    """
    Time-series cross-validator with expanding training window.
    
    Splits data into Train/Val/Test sets moving forward in time.
    
    Structure:
    Fold 1: |----Train----|--Val--|--Test--|
    Fold 2: |------Train------|--Val--|--Test--|
    Fold 3: |----------Train------|--Val--|--Test--|
    """
    def __init__(self, n_splits: int = 5, 
                 train_days: int = 252, 
                 val_days: int = 63, 
                 test_days: int = 21,
                 bars_per_day: int = 1440):
        """
        Args:
            n_splits: Number of folds
            train_days: Initial training window size in days
            val_days: Validation window size in days
            test_days: Test window size in days
            bars_per_day: Number of bars per day (default 1440 for 1min data)
        """
        self.n_splits = n_splits
        self.train_bars = train_days * bars_per_day
        self.val_bars = val_days * bars_per_day
        self.test_bars = test_days * bars_per_day
        
    def split(self, data: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate Train/Val/Test splits.
        
        Args:
            data: DataFrame to split (must be sorted by time)
            
        Yields:
            (train_df, val_df, test_df)
        """
        n_samples = len(data)
        
        # Calculate total required samples for one fold
        min_required = self.train_bars + self.val_bars + self.test_bars
        
        if n_samples < min_required:
            raise ValueError(f"Data length {n_samples} < min required {min_required}")
            
        # Calculate step size to fit n_splits
        # Last fold ends at n_samples
        # Last fold train end = n_samples - val - test
        # First fold train end = train_bars
        # Available room to slide = Last_train_end - First_train_end
        
        last_train_end = n_samples - self.val_bars - self.test_bars
        first_train_end = self.train_bars
        
        if last_train_end < first_train_end:
             # If data is too short for expanding window with fixed val/test, 
             # we might need to reduce n_splits or overlap.
             # For now, let's just use what we have, maybe reducing initial train.
             logger.warning("Data too short for requested configuration. Adjusting...")
             step_size = 0
        else:
            if self.n_splits > 1:
                step_size = (last_train_end - first_train_end) // (self.n_splits - 1)
            else:
                step_size = 0
            
        logger.info(f"Walk-Forward Split: {self.n_splits} folds, Step size: {step_size} bars")
        
        for i in range(self.n_splits):
            # Calculate indices
            train_end = first_train_end + (i * step_size)
            val_end = train_end + self.val_bars
            test_end = val_end + self.test_bars
            
            # Ensure we don't go out of bounds (shouldn't happen with above logic, but safety first)
            if test_end > n_samples:
                break
                
            # Slice data
            # Train: Start from 0 (Expanding)
            train_df = data.iloc[:train_end]
            val_df = data.iloc[train_end:val_end]
            test_df = data.iloc[val_end:test_end]
            
            yield train_df, val_df, test_df
            
    def get_indices(self, n_samples: int) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate indices instead of DataFrames.
        """
        min_required = self.train_bars + self.val_bars + self.test_bars
        if n_samples < min_required:
             raise ValueError(f"Data length {n_samples} < min required {min_required}")

        last_train_end = n_samples - self.val_bars - self.test_bars
        first_train_end = self.train_bars
        
        if self.n_splits > 1:
            step_size = (last_train_end - first_train_end) // (self.n_splits - 1)
        else:
            step_size = 0
            
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            train_end = first_train_end + (i * step_size)
            val_end = train_end + self.val_bars
            test_end = val_end + self.test_bars
            
            if test_end > n_samples:
                break
                
            yield (
                indices[:train_end],
                indices[train_end:val_end],
                indices[val_end:test_end]
            )

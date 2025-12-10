"""
Feature Normalization Module
Provides Robust and Z-Score normalization for trading features.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger("FeatureNormalizer")

class FeatureNormalizer:
    """
    Normalizes features using Robust Scaling (Median/IQR) or Z-Score (Mean/Std).
    Robust scaling is recommended for financial data with outliers.
    """
    def __init__(self, method: str = 'robust'):
        """
        Args:
            method: 'robust' (default) or 'zscore'
        """
        self.method = method
        self.params: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame):
        """
        Compute normalization parameters from data.
        """
        logger.info(f"Fitting normalizer with method: {self.method}")
        
        for col in df.columns:
            # Skip non-numeric columns
            if not np.issubdtype(df[col].dtype, np.number):
                continue
                
            if self.method == 'robust':
                q25 = df[col].quantile(0.25)
                q75 = df[col].quantile(0.75)
                iqr = q75 - q25
                median = df[col].median()
                
                self.params[col] = {
                    'center': median,
                    'scale': iqr
                }
            elif self.method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                
                self.params[col] = {
                    'center': mean,
                    'scale': std
                }
                
        self.is_fitted = True
        logger.info(f"Fitted {len(self.params)} columns")
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization to data.
        """
        if not self.is_fitted:
            logger.warning("Normalizer not fitted, returning original data")
            return df
            
        result = df.copy()
        
        for col in df.columns:
            if col in self.params:
                p = self.params[col]
                scale = p['scale']
                
                # Avoid division by zero
                if scale == 0:
                    scale = 1.0
                    
                result[col] = (df[col] - p['center']) / scale
                
                # For robust scaling, we might still have extreme outliers
                # Optional: Clip to reasonable range (e.g., +/- 10)
                if self.method == 'robust':
                    result[col] = result[col].clip(-10, 10)
                    
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        """
        self.fit(df)
        return self.transform(df)
    
    def save_params(self, path: str):
        """Save parameters to file (e.g., JSON/Pickle)"""
        import json
        with open(path, 'w') as f:
            json.dump({'method': self.method, 'params': self.params}, f)
            
    def load_params(self, path: str):
        """Load parameters from file"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.method = data['method']
            self.params = data['params']
            self.is_fitted = True

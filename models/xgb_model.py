import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Tuple, Optional

class XGBoostModel:
    def __init__(self, model_path: str = "models/xgb_direction.json"):
        self.model_path = model_path
        self.model = None
        self.version = "1.0.0"
        self.metadata = {
            "created_at": "2023-10-27",
            "features": ["ret_1", "ret_5", "vol_20", "ma_20", "rsi"],
            "target": "direction_3class"
        }
        self.params = {
            'objective': 'multi:softprob',
            'num_class': 3, # Down, Neutral, Up
            'eval_metric': 'mlogloss',
            'eta': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'nthread': -1
        }

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from OHLCV data.
        Assumes df has columns: open, high, low, close, volume
        """
        df = df.copy()
        
        # Returns
        df['ret_1'] = df['close'].pct_change(1)
        df['ret_5'] = df['close'].pct_change(5)
        
        # Volatility
        df['vol_20'] = df['ret_1'].rolling(20).std()
        
        # Simple MA
        df['ma_20'] = df['close'].rolling(20).mean()
        df['dist_ma_20'] = (df['close'] - df['ma_20']) / df['ma_20']
        
        # RSI (Simplified)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df.dropna()

    def prepare_targets(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """
        Generate targets: 0=Down, 1=Neutral, 2=Up
        Threshold based on ATR or fixed %.
        """
        future_ret = df['close'].shift(-horizon) / df['close'] - 1
        threshold = 0.0005 # 0.05%
        
        targets = pd.Series(1, index=df.index) # Default Neutral
        targets[future_ret > threshold] = 2 # Up
        targets[future_ret < -threshold] = 0 # Down
        
        return targets.dropna()

    def train(self, df: pd.DataFrame):
        """Train the model."""
        # Feature Engineering
        data = self.prepare_features(df)
        targets = self.prepare_targets(df)
        
        # Align using inner join to guarantee matching indices
        # Rename columns to avoid collisions if any
        data.columns = [f"feat_{c}" for c in data.columns]
        
        # Merge
        aligned = data.join(targets.rename("target"), how="inner")
        
        X = aligned[[c for c in aligned.columns if c.startswith("feat_")]]
        y = aligned["target"]
        
        print(f"DEBUG: Training on {len(X)} samples. Features: {X.shape}, Targets: {y.shape}")
        
        if X.empty or y.empty:
            print("WARNING: Training data is empty!")
            return

        # Ensure strict alignment (XGBoost is sensitive to this)
        if len(X) != len(y):
            print(f"Warning: Alignment mismatch X={len(X)}, y={len(y)}")
            return

        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Train
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=100,
            xgb_model=self.model # Incremental training if model exists
        )

    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """Predict probabilities for the latest candle."""
        if self.model is None:
            self.load()
            
        features = self.prepare_features(df)
        if features.empty:
            return {'down': 0, 'neutral': 1, 'up': 0}
            
        latest = features.iloc[[-1]]
        dtest = xgb.DMatrix(latest)
        probs = self.model.predict(dtest)[0]
        
        return {
            'down': float(probs[0]),
            'neutral': float(probs[1]),
            'up': float(probs[2])
        }

    def save(self):
        if self.model is None:
            print("Warning: Model is None, skipping save.")
            return

        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        self.model.save_model(self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

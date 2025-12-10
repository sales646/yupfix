import pandas as pd
import numpy as np
from arch import arch_model

class FeatureEngineer:
    """
    Advanced Feature Engineering for Hybrid ML-RL Strategy.
    Calculates:
    - GARCH Volatility (Stochastic Volatility)
    - Technical Indicators (RSI, MACD, EMA)
    - Returns & Momentum
    """
    
    def __init__(self):
        pass
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Returns
        df['ret_1'] = df['close'].pct_change().fillna(0)
        df['ret_5'] = df['close'].pct_change(5).fillna(0)
        
        # 2. GARCH Volatility (The "Heston-lite" approach)
        # We model the variance of returns
        # Rescale returns for numerical stability (GARCH hates small numbers)
        returns_scaled = df['ret_1'] * 100 
        
        try:
            # GARCH(1,1) is standard for financial time series
            model = arch_model(returns_scaled, vol='Garch', p=1, q=1, rescale=False)
            # We fit on the whole history (in production, use rolling window!)
            # For speed in this prototype, we fit once.
            res = model.fit(disp='off')
            df['garch_vol'] = res.conditional_volatility
        except Exception as e:
            print(f"GARCH Failed: {e}, using rolling std")
            df['garch_vol'] = df['ret_1'].rolling(20).std() * 100
            
        # 3. Technical Indicators
        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        # MACD (12, 26, 9)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd'] = macd - signal # Histogram
        
        # Distance from MA 200 (Trend)
        ma_200 = df['close'].rolling(window=200).mean()
        df['dist_ma_200'] = (df['close'] - ma_200) / ma_200
        df['dist_ma_200'] = df['dist_ma_200'].fillna(0)
        
        # Rename for Env compatibility
        df['dist_ma_20'] = df['dist_ma_200'] # Reuse slot for long trend
        
        # 4. ATR (Average True Range) - 14 period for position sizing
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        df['atr'] = df['atr'].fillna(df['atr'].mean())
        
        # 5. Time-based features (for session awareness)
        if df.index.tz is None:
            # Assume UTC if no timezone
            df.index = pd.to_datetime(df.index, utc=True)
        
        df['hour'] = df.index.hour
        # Sine/Cosine encoding preserves cyclical nature of time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Session flags (0=Asia, 1=London, 2=NY, 3=Overlap)
        def get_session(hour):
            if 13 <= hour < 17:
                return 3  # London-NY Overlap (highest liquidity)
            elif 8 <= hour < 17:
                return 1  # London
            elif (13 <= hour < 24) or (0 <= hour < 8):
                return 2  # NY/Asia overnight
            else:
                return 0  # Asia
        
        df['session'] = df['hour'].apply(get_session)
        
        # Normalize features (Critical for RL/PPO)
        # We'll do simple z-score or scaling here or let PPO normalize
        # Let's keep raw for now, PPO usually handles normalization if configured.
        
        return df.dropna()

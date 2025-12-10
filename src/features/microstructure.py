"""
Microstructure feature engineering - 21 features
"""
import pandas as pd
import numpy as np
import logging
from typing import List
from .normalization import FeatureNormalizer
from ..contracts.data import FeatureRow

logger = logging.getLogger(__name__)

class MicrostructureFeatureEngineer:
    """Extract 21 market microstructure features from tick-aggregated data."""
    
    def __init__(self, bar_seconds: int = 1, normalize: bool = True):
        self.bar_seconds = bar_seconds
        self.normalize = normalize
        self.normalizer = FeatureNormalizer(method='robust')
        
    def _window(self, seconds: int) -> int:
        """Convert seconds to number of bars"""
        return max(1, int(seconds // self.bar_seconds))

    def fit(self, df: pd.DataFrame):
        """Fit the normalizer on training data"""
        if self.normalize:
            # Compute features first (without normalization)
            features = self._compute_raw_features(df)
            # Fit normalizer
            self.normalizer.fit(features)
            
    def save_scaler(self, path: str):
        self.normalizer.save_params(path)
        
    def load_scaler(self, path: str):
        self.normalizer.load_params(path)
        
    def _compute_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal method to compute features without normalization"""
        if df.empty:
            return df
            
        features = pd.DataFrame(index=df.index)
        
        # 1. Order Flow Imbalance (OFI)
        if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
            total_vol = df['volume'].replace(0, 1)
            features['ofi'] = (df['buy_volume'] - df['sell_volume']) / total_vol
            # OFI windows in SECONDS (not bars)
            ofi_windows_seconds = [100, 500, 1000]
            for w_sec in ofi_windows_seconds:
                w_bars = self._window(w_sec)
                # Use seconds in column name to match FeatureRow contract
                features[f'ofi_{w_sec}s'] = features['ofi'].rolling(window=w_bars).sum()
        
        # 2. Amihud Illiquidity
        if 'close' in df.columns and 'volume' in df.columns:
            returns = df['close'].pct_change().abs()
            features['amihud_illiq'] = (returns / df['volume'].replace(0, np.nan)) * 1e6
            features['amihud_illiq'] = features['amihud_illiq'].fillna(0)
            
        # 3. Spread Features
        if 'spread_max' in df.columns and 'spread_avg' in df.columns:
            features['spread_volatility'] = df['spread_max'] - df['spread_avg']
            if 'close' in df.columns:
                features['effective_spread_pct'] = (df['spread_avg'] / df['close']) * 100
                
        # 4. Tick Velocity
        if 'tick_count' in df.columns:
            features['tick_velocity'] = df['tick_count']
            if 'volume' in df.columns:
                features['trade_size_avg'] = df['volume'] / df['tick_count'].replace(0, 1)
                
        # 5. Flow Toxicity
        if 'ofi' in features.columns:
            features['flow_toxicity'] = features['ofi'].rolling(window=10).std()

        # 6. Microprice Deviation
        if all(c in df.columns for c in ['bid_close', 'ask_close', 'buy_volume', 'sell_volume']):
            bid, ask = df['bid_close'], df['ask_close']
            ask_vol, bid_vol = df['buy_volume'], df['sell_volume']
            total_vol = ask_vol + bid_vol
            microprice = (bid * ask_vol + ask * bid_vol) / total_vol.replace(0, 1)
            mid = (bid + ask) / 2
            microprice = microprice.where(total_vol > 0, mid)
            features['microprice_deviation'] = df['close'] - microprice

        # 7. VWAP Deviation (1 hour window)
        if 'close' in df.columns and 'volume' in df.columns:
            w_vwap = self._window(3600)
            pv = df['close'] * df['volume']
            vwap = pv.rolling(window=w_vwap).sum() / df['volume'].rolling(window=w_vwap).sum().replace(0, 1)
            features['vwap_deviation'] = df['close'] - vwap

        # 8. Bollinger Bands (20 min window)
        if 'close' in df.columns:
            w_bb = self._window(20 * 60)
            rolling_mean = df['close'].rolling(window=w_bb).mean()
            rolling_std = df['close'].rolling(window=w_bb).std()
            upper_band = rolling_mean + (rolling_std * 2)
            lower_band = rolling_mean - (rolling_std * 2)
            features['bollinger_percent_b'] = (df['close'] - lower_band) / (upper_band - lower_band).replace(0, 1)
            features['bollinger_bandwidth'] = (upper_band - lower_band) / rolling_mean

        # 9. RSI (14 min window)
        if 'close' in df.columns:
            w_rsi = self._window(14 * 60)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=w_rsi).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w_rsi).mean()
            rs = gain / loss.replace(0, 1)
            features['rsi'] = 100 - (100 / (1 + rs))
            
            price_trend = df['close'].rolling(window=20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
            rsi_trend = features['rsi'].rolling(window=20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
            features['rsi_divergence'] = (price_trend != rsi_trend).astype(int)
            
        # 10. Mean Reversion Score
        if 'bollinger_percent_b' in features.columns and 'rsi' in features.columns:
            features['mean_reversion_score'] = (
                (features['bollinger_percent_b'] < 0.2).astype(float) * 0.5 + 
                (features['rsi'] < 30).astype(float) * 0.5
            ) - (
                (features['bollinger_percent_b'] > 0.8).astype(float) * 0.5 +
                (features['rsi'] > 70).astype(float) * 0.5
            )

        # 11. Bid-Ask Imbalance (NEW)
        if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
            features['bid_ask_imbalance'] = df['buy_volume'] / (df['buy_volume'] + df['sell_volume']).replace(0, 1)
            
        # 12. Volatility Regime (NEW)
        if 'close' in df.columns:
            w_short = self._window(60)
            w_long = self._window(600)
            vol_short = df['close'].pct_change().rolling(w_short).std()
            vol_long = df['close'].pct_change().rolling(w_long).std()
            features['vol_regime'] = vol_short / vol_long.replace(0, 1)
            
        # 13. Liquidity Shock (NEW)
        if 'spread_max' in df.columns:
            w_spread = self._window(300)
            spread_ma = df['spread_max'].rolling(w_spread).mean()
            features['liquidity_shock'] = df['spread_max'] / spread_ma.replace(0, 1)
            
        # 14. Price Entropy (NEW)
        if 'close' in df.columns:
            def calculate_entropy(x):
                if len(x) < 2:
                    return 0.0
                hist, _ = np.histogram(x, bins=10, density=True)
                hist = hist[hist > 0]
                if len(hist) == 0:
                    return 0.0
                return -np.sum(hist * np.log(hist + 1e-10))
                
            w_entropy = self._window(100)
            features['price_entropy'] = df['close'].pct_change().rolling(w_entropy).apply(
                calculate_entropy, raw=True
            )

        # Clip extreme values (5th/95th percentile)
        for col in features.columns:
            lower = features[col].quantile(0.05)
            upper = features[col].quantile(0.95)
            features[col] = features[col].clip(lower, upper)
            
        return features

    def compute_features(self, df: pd.DataFrame) -> List[FeatureRow]:
        features = self._compute_raw_features(df)
        
        # Normalization
        if self.normalize:
            if self.normalizer.is_fitted:
                features = self.normalizer.transform(features)
            else:
                logger.warning("Normalizer not fitted! Performing online fitting (might leak info). Call fit() first.")
                features = self.normalizer.fit_transform(features)
        
        # Fill NaNs (e.g. from rolling windows) with 0 (mean/median after normalization)
        features.fillna(0, inplace=True)
        
        # Convert to List[FeatureRow]
        # Ensure timestamp is a column
        features['timestamp'] = features.index
        
        # Handle potential missing columns by filling with 0 or appropriate defaults
        # This is critical because FeatureRow is strict
        required_cols = FeatureRow.__fields__.keys()
        for col in required_cols:
            if col not in features.columns:
                features[col] = 0.0 # Default fill
                
        # Convert to list of contracts
        contract_list = []
        for _, row in features.iterrows():
            # Convert row to dict and create FeatureRow
            # We need to handle Timestamp conversion if needed, but pydantic handles datetime
            data_dict = row.to_dict()
            try:
                contract_list.append(FeatureRow(**data_dict))
            except Exception as e:
                logger.error(f"Validation failed for row {row.name}: {e}")
                continue
                
        return contract_list

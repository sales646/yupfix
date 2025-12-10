"""
Global HMM for market regime detection
"""
import numpy as np
import pandas as pd
import joblib
import logging

logger = logging.getLogger(__name__)

# Try to import hmmlearn, fallback to dummy if not available
try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.decomposition import PCA
    HMM_AVAILABLE = True
except ImportError:
    logger.warning("hmmlearn not installed - using dummy HMM")
    HMM_AVAILABLE = False


class GlobalHMM:
    """
    Hidden Markov Model for detecting global market regimes.
    Falls back to simple volatility-based regime if hmmlearn unavailable.
    """
    STATE_NAMES = {0: "STABLE", 1: "UNCERTAIN", 2: "CRISIS"}
    
    def __init__(self, n_states: int = 3, window_seconds: int = 3600, 
                 bar_seconds: int = 1):
        self.n_states = n_states
        self.window_seconds = window_seconds
        self.bar_seconds = bar_seconds
        self.window = max(1, window_seconds // bar_seconds)
        self.is_fitted = False
        self.use_real_hmm = HMM_AVAILABLE
        
        if HMM_AVAILABLE:
            self.hmm = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            self.pca = PCA(n_components=1)
        else:
            self.hmm = None
            self.pca = None
            
        self.state_mapping = np.array([0, 1, 2])
        self.vol_thresholds = None  # For fallback method
    
    def fit(self, multi_asset_returns: pd.DataFrame):
        """Fit HMM or fallback volatility model."""
        if self.use_real_hmm:
            self._fit_real_hmm(multi_asset_returns)
        else:
            self._fit_volatility_fallback(multi_asset_returns)
        
        self.is_fitted = True
        logger.info(f"HMM fitted (real={self.use_real_hmm})")
    
    def _fit_real_hmm(self, multi_asset_returns: pd.DataFrame):
        """Fit actual HMM."""
        self.pca.fit(multi_asset_returns.dropna())
        features, _ = self._extract_features(multi_asset_returns)
        self.hmm.fit(features)
        
        # Sort states by volatility
        state_vols = [self.hmm.means_[s, 0] for s in range(self.n_states)]
        self.state_mapping = np.argsort(state_vols)
    
    def _fit_volatility_fallback(self, multi_asset_returns: pd.DataFrame):
        """Simple volatility-based regime detection."""
        volatility = multi_asset_returns.rolling(self.window).std().mean(axis=1).dropna()
        
        # Compute tercile thresholds
        self.vol_thresholds = [
            volatility.quantile(0.33),
            volatility.quantile(0.67)
        ]
        logger.info(f"Volatility thresholds: {self.vol_thresholds}")
    
    def _extract_features(self, multi_asset_returns: pd.DataFrame):
        """Extract features for real HMM."""
        volatility = multi_asset_returns.rolling(self.window).std().mean(axis=1)
        
        # Simplified correlation (EMA-based)
        alpha = 0.001
        returns = multi_asset_returns.values
        n_assets = returns.shape[1]
        
        corrs = []
        ema_r = np.zeros(n_assets)
        ema_r2 = np.zeros(n_assets)
        ema_rr = np.zeros((n_assets, n_assets))
        
        for i in range(len(returns)):
            r = returns[i]
            if np.isnan(r).any():
                corrs.append(np.nan)
                continue
            
            ema_r = alpha * r + (1 - alpha) * ema_r
            ema_r2 = alpha * r**2 + (1 - alpha) * ema_r2
            ema_rr = alpha * np.outer(r, r) + (1 - alpha) * ema_rr
            
            var = ema_r2 - ema_r**2
            cov = ema_rr - np.outer(ema_r, ema_r)
            std = np.sqrt(np.maximum(var, 1e-8))
            corr = cov / np.outer(std, std)
            
            mask = np.triu(np.ones((n_assets, n_assets), dtype=bool), k=1)
            corrs.append(corr[mask].mean())
        
        correlation = pd.Series(corrs, index=multi_asset_returns.index)
        
        # Market factor
        clean_data = multi_asset_returns.fillna(0)
        market_factor = pd.Series(
            self.pca.transform(clean_data)[:, 0],
            index=multi_asset_returns.index
        )
        
        features = pd.DataFrame({
            'volatility': volatility,
            'correlation': correlation,
            'market_factor': market_factor
        }).dropna()
        
        return features.values, features.index
    
    def predict(self, multi_asset_returns: pd.DataFrame) -> np.ndarray:
        """Predict regime states."""
        if not self.is_fitted:
            raise RuntimeError("HMM not fitted. Call fit() first.")
        
        if self.use_real_hmm:
            features, _ = self._extract_features(multi_asset_returns)
            raw_states = self.hmm.predict(features)
            return np.array([self.state_mapping[s] for s in raw_states])
        else:
            return self._predict_volatility_fallback(multi_asset_returns)
    
    def _predict_volatility_fallback(self, multi_asset_returns: pd.DataFrame) -> np.ndarray:
        """Predict using volatility thresholds."""
        volatility = multi_asset_returns.rolling(self.window).std().mean(axis=1)
        
        states = np.ones(len(volatility), dtype=int)  # Default: UNCERTAIN
        states[volatility < self.vol_thresholds[0]] = 0  # STABLE
        states[volatility > self.vol_thresholds[1]] = 2  # CRISIS
        
        return states
    
    def predict_proba(self, multi_asset_returns: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities."""
        if not self.is_fitted:
            raise RuntimeError("HMM not fitted. Call fit() first.")
        
        if self.use_real_hmm:
            features, _ = self._extract_features(multi_asset_returns)
            raw_proba = self.hmm.predict_proba(features)
            return raw_proba[:, self.state_mapping]
        else:
            # Fallback: hard probabilities
            states = self._predict_volatility_fallback(multi_asset_returns)
            proba = np.zeros((len(states), self.n_states))
            for i, s in enumerate(states):
                proba[i, s] = 1.0
            return proba

    def predict_proba_with_index(self, multi_asset_returns: pd.DataFrame) -> tuple:
        """Predict regime probabilities and return aligned index."""
        if not self.is_fitted:
            raise RuntimeError("HMM not fitted. Call fit() first.")
        
        if self.use_real_hmm:
            features, valid_index = self._extract_features(multi_asset_returns)
            raw_proba = self.hmm.predict_proba(features)
            return raw_proba[:, self.state_mapping], valid_index
        else:
            # Fallback
            states = self._predict_volatility_fallback(multi_asset_returns)
            proba = np.zeros((len(states), self.n_states))
            for i, s in enumerate(states):
                proba[i, s] = 1.0
            # Fallback uses rolling().dropna(), so index is also reduced
            # We need to replicate that logic to get index
            # _predict_volatility_fallback returns numpy array, losing index
            # Let's just return the last N indices
            return proba, multi_asset_returns.index[-len(proba):]
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get state transition probabilities."""
        if not self.is_fitted:
            raise RuntimeError("HMM not fitted.")
        
        if self.use_real_hmm:
            raw_trans = self.hmm.transmat_
            return raw_trans[self.state_mapping][:, self.state_mapping]
        else:
            # Dummy transition matrix
            return np.array([
                [0.95, 0.04, 0.01],
                [0.10, 0.80, 0.10],
                [0.05, 0.15, 0.80]
            ])
    
    def save(self, path: str):
        """Save fitted model."""
        joblib.dump({
            'hmm': self.hmm,
            'pca': self.pca,
            'state_mapping': self.state_mapping,
            'vol_thresholds': self.vol_thresholds,
            'use_real_hmm': self.use_real_hmm,
            'config': {
                'n_states': self.n_states,
                'window_seconds': self.window_seconds,
                'bar_seconds': self.bar_seconds
            }
        }, path)
    
    def load(self, path: str):
        """Load fitted model."""
        data = joblib.load(path)
        self.hmm = data['hmm']
        self.pca = data['pca']
        self.state_mapping = data['state_mapping']
        self.vol_thresholds = data.get('vol_thresholds')
        self.use_real_hmm = data.get('use_real_hmm', True)
        self.n_states = data['config']['n_states']
        self.window_seconds = data['config']['window_seconds']
        self.bar_seconds = data['config']['bar_seconds']
        self.window = max(1, self.window_seconds // self.bar_seconds)
        self.is_fitted = True

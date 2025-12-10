"""
Yup250 Training Pipeline
Full integration of data loading, feature engineering, HMM, and Mamba model
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

from .features.microstructure import MicrostructureFeatureEngineer
from .models.global_hmm import GlobalHMM
from .models.mamba_full import MambaTrader
from .models.mamba_ensemble import MambaEnsemble
from .data.validator import DataValidator

logger = logging.getLogger(__name__)

class Yup250Pipeline:
    """
    Complete training pipeline for Yup250 Mamba trading system.
    
    Pipeline flow:
    1. Load multi-asset OHLCV data
    2. Compute microstructure features
    3. Fit Global HMM for regime detection
    4. Create Mamba model
    5. Train with multi-scale targets
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Data config
        self.symbols = config['data']['symbols']
        self.sequence_length = config['data']['sequence_length']
        self.bar_seconds = config['data']['bar_seconds']
        
        # Model config
        self.model_config = config['model']
        
        # Initialize feature engineer
        self.feature_engineer = MicrostructureFeatureEngineer(
            bar_seconds=self.bar_seconds
        )
        
        # Initialize HMM
        hmm_config = config['hmm']
        self.hmm = GlobalHMM(
            n_states=hmm_config['n_states'],
            window_seconds=hmm_config['window_seconds'],
            bar_seconds=self.bar_seconds
        )
        
        self.model = None
        self.is_fitted = False
        self.scalers = {}
        
    def load_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load multi-asset precomputed feature data across timeframes.
        Merges 1min, 5min, 15min, 1hour features into a single DataFrame per symbol.
        
        Args:
            data_path: Base path to data directory (e.g. data/features)
            
        Returns:
            Dict mapping symbol to DataFrame (aligned 1min grid)
        """
        base_dir = Path(data_path).parent # Assuming input is data/features/1min, we want data/features
        if base_dir.name != 'features':
             # Fallback if user passed "data/features" directly
             if Path(data_path).name == 'features':
                 base_dir = Path(data_path)
             else:
                 # Assume structure is .../features/1min
                 base_dir = Path(data_path).parent
                 
        timeframes = ['1min', '5min', '15min', '1hour']
        data = {}
        
        for symbol in self.symbols:
            logger.info(f"Loading multi-timeframe data for {symbol}...")
            merged_df = None
            
            for tf in timeframes:
                file_path = base_dir / tf / f"{symbol}.parquet"
                if not file_path.exists():
                    logger.warning(f"Data file not found: {file_path}")
                    continue
                    
                df = pd.read_parquet(file_path)
                
                # Set timestamp as index
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                
                # Rename columns with suffix (except for 1min which is base, but let's be consistent)
                # Actually, 1min cols usually don't have suffix in single-TF models.
                # But for 100-feature model, we want distinction.
                # Let's suffix ALL columns to be safe and clear.
                suffix = f"_{tf}"
                df.columns = [f"{c}{suffix}" for c in df.columns]
                
                if merged_df is None:
                    # Base dataframe (1min)
                    merged_df = df
                else:
                    # Merge onto base (left join)
                    # We want to align 5min/15min/1h data to 1min timestamps.
                    # Since timestamps are BAR-END, a 10:05 5min bar covers 10:00-10:05.
                    # At 10:01 (1min bar), we should ideally see the *developing* 5min bar?
                    # NO. That would be lookahead if we use the final 10:05 value.
                    # We must use the *previous* completed bar (10:00).
                    # Standard merge + ffill does exactly this:
                    # 10:00 5min exists -> 10:00 1min gets it.
                    # 10:01 1min -> NaN (since 5min has no 10:01) -> ffill takes 10:00.
                    # This is SAFE.
                    merged_df = merged_df.join(df, how='left')
            
            if merged_df is not None:
                # Forward fill to propagate lower frequency data
                merged_df.ffill(inplace=True)
                
                # Drop any remaining NaNs (start of data)
                merged_df.dropna(inplace=True)
                
                data[symbol] = merged_df
                logger.info(f"Loaded {symbol}: {len(merged_df)} rows, {len(merged_df.columns)} features")
            else:
                logger.warning(f"No data found for {symbol}")
                
        if not data:
            raise ValueError(f"No data found in {base_dir}")
            
        return data
    
    def get_regime_probs(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Get regime probabilities from HMM.
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with regime probabilities (N, n_states), indexed by timestamp
        """
        if not self.hmm.is_fitted:
            # If not fitted, return empty or raise?
            # Should be fitted if we are here.
            # But if we are in training and just loaded data, we fit it in fit_hmm called separately.
            # If we call this inside prepare_features, we assume fit_hmm was called.
            if not self.is_fitted and not self.hmm.is_fitted:
                 # Auto-fit if needed (e.g. inference time?)
                 # No, training pipeline controls fitting.
                 logger.warning("HMM not fitted. Returning zeros.")
                 return None
            
        # Combine returns
        returns_list = []
        common_index = None
        
        for symbol, df in data.items():
            # Use 1min close for HMM
            col = 'close_1min' if 'close_1min' in df.columns else 'close'
            if col in df.columns:
                returns = df[col].pct_change()
                returns_list.append(returns)
                
                if common_index is None:
                    common_index = returns.index
                else:
                    common_index = common_index.intersection(returns.index)
        
        if common_index is None or len(common_index) == 0:
            logger.warning("No common index for HMM")
            return None
            
        # Create multi-asset returns DataFrame
        multi_asset_returns = pd.DataFrame(
            {f"asset_{i}": ret.loc[common_index] for i, ret in enumerate(returns_list)},
            index=common_index
        )
        
        # Drop NaNs (pct_change introduces NaNs at start)
        multi_asset_returns.dropna(inplace=True)
        
        if multi_asset_returns.empty:
            logger.warning("Multi-asset returns empty after dropping NaNs")
            return None
            
        # Update common_index to match cleaned returns
        common_index = multi_asset_returns.index
        
        logger.info(f"HMM input shape: {multi_asset_returns.shape}")
        logger.info(f"Common index length: {len(common_index)}")
        
        # Predict with aligned index
        probs, aligned_index = self.hmm.predict_proba_with_index(multi_asset_returns)
        
        logger.info(f"HMM probs shape: {probs.shape}")
        logger.info(f"Aligned index length: {len(aligned_index)}")
        
        # Return as DataFrame
        prob_cols = [f"regime_prob_{i}" for i in range(probs.shape[1])]
        prob_df = pd.DataFrame(probs, columns=prob_cols, index=aligned_index)
        
        return prob_df

    def prepare_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Prepare features for all symbols.
        Adds HMM regime features to the loaded multi-timeframe features.
        
        Args:
            data: Feature DataFrames per symbol (multi-TF merged)
            
        Returns:
            Features per symbol (validated, N features + 4 HMM = 100)
        """
        features = {}
        
        # Calculate HMM regime probabilities
        regime_df = self.get_regime_probs(data)
        
        for symbol, df in data.items():
            logger.info(f"Preparing features for {symbol}...")
            
            # Ensure timestamp is index
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            
            feat_df = df.copy()
            
            # Add HMM features
            if regime_df is not None:
                # Join on index (left join to keep all rows, fill missing with 0)
                feat_df = feat_df.join(regime_df, how='left')
                
                # Fill NaNs (e.g. rows not in common index)
                for c in regime_df.columns:
                    feat_df[c] = feat_df[c].fillna(0.0)
                
                # Add regime state (argmax)
                # We can compute it from probs
                prob_cols = regime_df.columns
                feat_df['regime_state'] = feat_df[prob_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(float).fillna(0)
            else:
                # Add dummy columns if HMM failed
                for i in range(3): # Assume 3 states
                    feat_df[f'regime_prob_{i}'] = 0.0
                feat_df['regime_state'] = 0.0
            
            features[symbol] = feat_df
            logger.info(f"{symbol} features: {feat_df.shape}")
            
        return features


    def create_dataloader(self, features: Dict[str, pd.DataFrame], 
                          labels: Dict[str, pd.DataFrame],
                          batch_size: int, shuffle: bool = True,
                          stride: int = None) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for training/validation.
        
        Args:
            features: Dict of feature DataFrames
            labels: Dict of label DataFrames
            batch_size: Batch size
            shuffle: Whether to shuffle
            stride: Sample every Nth position (default from config, or 5)
            
        Returns:
            DataLoader
        """
        # Default stride from config or use 5 for efficiency
        if stride is None:
            stride = self.config.get('training', {}).get('stride', 5)
        
        dataset = Yup250Dataset(
            features=features,
            labels=labels,
            sequence_length=self.sequence_length,
            symbols=self.symbols,
            stride=stride
        )
        
        # Get num_workers from config (default 0 for Windows compatibility)
        num_workers = self.config.get('training', {}).get('num_workers', 0)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False
        )

    def create_model(self, use_ensemble: bool = False, n_models: int = 3) -> torch.nn.Module:
        """
        Create Mamba model.
        
        Args:
            use_ensemble: Whether to use ensemble
            n_models: Number of models in ensemble
            
        Returns:
            Model instance
        """
        if use_ensemble:
            logger.info(f"Creating ensemble with {n_models} models...")
            model = MambaEnsemble(
                model_class=MambaTrader,
                model_kwargs=self.model_config,
                n_models=n_models
            )
        else:
            logger.info("Creating single Mamba model...")
            # Filter out keys not accepted by MambaTrader
            # Based on error: dropout is not accepted
            model_kwargs = self.model_config.copy()
            if 'dropout' in model_kwargs:
                del model_kwargs['dropout']
            if 'use_multi_label' in model_kwargs:
                 del model_kwargs['use_multi_label'] # Assuming this might also be extra if not used in init
                 
            model = MambaTrader(**model_kwargs)
        
        self.model = model
        self.is_fitted = True
        
        return model

    def fit_hmm(self, data: Dict[str, pd.DataFrame]):
        """
        Fit Global HMM on multi-asset returns.
        
        Args:
            data: OHLCV data per symbol
        """
        logger.info("Fitting Global HMM...")
        
        # Combine returns from all symbols
        returns_list = []
        common_index = None
        
        for symbol, df in data.items():
            # Use 1min close for HMM fitting
            col = 'close_1min' if 'close_1min' in df.columns else 'close'
            if col in df.columns:
                returns = df[col].pct_change()
                returns_list.append(returns)
                
                if common_index is None:
                    common_index = returns.index
                else:
                    # Get intersection of indices
                    common_index = common_index.intersection(returns.index)
        
        if common_index is None or len(common_index) == 0:
            logger.warning("No common index for HMM fitting")
            return

        # Create multi-asset returns DataFrame
        multi_asset_returns = pd.DataFrame(
            {f"asset_{i}": ret.loc[common_index] for i, ret in enumerate(returns_list)},
            index=common_index
        )
        
        # Fit HMM
        self.hmm.fit(multi_asset_returns)
        logger.info("HMM fitted successfully")

    def save_scalers(self, save_dir: Path):
        """Save feature scalers for all symbols"""
        import json
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        scaler_path = save_dir / "scalers.json"
        with open(scaler_path, 'w') as f:
            json.dump(self.scalers, f, indent=2)
        logger.info(f"Saved scalers to {scaler_path}")


class Yup250Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Yup250 training.
    
    Returns sequences of (features, labels) for ALL assets simultaneously.
    Output shape: (N_assets, Seq_len, Features)
    
    Args:
        stride: Sample every Nth position. stride=5 means 5x fewer samples.
                This maintains coverage of the full data range while reducing
                training time. Default=1 (no striding).
    """
    
    def __init__(self, features: Dict[str, pd.DataFrame], 
                 labels: Dict[str, pd.DataFrame],
                 sequence_length: int, symbols: List[str],
                 stride: int = 1):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.symbols = symbols
        self.stride = max(1, stride)
        
        # Assume all aligned, use first symbol for length
        if not symbols:
            self.total_positions = 0
        else:
            first_sym = symbols[0]
            self.total_positions = len(features[first_sym]) - sequence_length
        
        # Create strided indices
        self.indices = list(range(0, self.total_positions, self.stride))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map idx to actual data position using strided indices
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        
        X_list = []
        y_list = []
        
        for symbol in self.symbols:
            # Get features
            feat_seq = self.features[symbol].iloc[start_idx:end_idx].values
            X_list.append(feat_seq)
            
            # Get labels
            label_seq = self.labels[symbol].iloc[start_idx:end_idx].values
            y_list.append(label_seq)
            
        # Stack to (N, L, F)
        X = torch.from_numpy(np.stack(X_list)).float()
        y = torch.from_numpy(np.stack(y_list))
        
        # Return symbols as a list (will be collated into list of lists or similar)
        # Actually, standard collate might fail on list of lists if not careful?
        # Let's return a simple string or just the list.
        # Trainer doesn't use it, so let's return the list.
        return X, y, self.symbols

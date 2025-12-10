# src/contracts/pipeline_validator.py
"""
Validates ENTIRE pipeline before training starts.
Catches ALL mismatches between components.
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class PipelineValidator:
    """One-time validation before training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_all(self, 
                     features: Dict[str, pd.DataFrame],
                     labels: Dict[str, pd.DataFrame],
                     model: torch.nn.Module) -> bool:
        """
        Run ALL validations. Fails fast with clear errors.
        """
        print("\n" + "="*60)
        print("PIPELINE VALIDATION")
        print("="*60)
        
        self.errors = []
        self.warnings = []
        
        # 1. Config internal consistency
        print("\n[1/6] Validating config...")
        self._validate_config()
        
        # 2. Features
        print("[2/6] Validating features...")
        self._validate_features(features)
        
        # 3. Labels
        print("[3/6] Validating labels...")
        self._validate_labels(labels)
        
        # 4. Feature ↔ Label alignment
        print("[4/6] Validating alignment...")
        self._validate_alignment(features, labels)
        
        # 5. Model architecture
        print("[5/6] Validating model...")
        self._validate_model(model)
        
        # 6. End-to-end dry run
        print("[6/6] Running dry run...")
        self._validate_dry_run(features, labels, model)
        
        # Results
        print("\n" + "="*60)
        
        if self.warnings:
            print(f"WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"   * {w}")
        
        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for e in self.errors:
                print(f"   * {e}")
            print("\nPIPELINE VALIDATION FAILED")
            print("="*60 + "\n")
            raise ValueError(f"Pipeline validation failed with {len(self.errors)} errors")
        
        print("\nPIPELINE VALIDATION PASSED")
        print("="*60 + "\n")
        return True
    
    def _validate_config(self):
        """Config internal consistency"""
        model_cfg = self.config.get('model', {})
        data_cfg = self.config.get('data', {})
        
        # d_input must match expected features
        # d_input must match expected features
        # NOTE: Adjusted to 100 based on restored multi-timeframe + HMM features
        if model_cfg.get('d_input') != 100:
            self.warnings.append(
                f"d_input={model_cfg.get('d_input')}, expected 100 technical features"
            )
        
        # n_assets must match symbols
        n_symbols = len(data_cfg.get('symbols', []))
        if model_cfg.get('n_assets') != n_symbols:
            self.errors.append(
                f"n_assets={model_cfg.get('n_assets')} != {n_symbols} symbols in config"
            )
        
        # sequence_length sanity
        seq_len = data_cfg.get('sequence_length', 0)
        if seq_len > 50000:
            self.warnings.append(f"sequence_length={seq_len} is very long, may cause OOM")
        if seq_len < 100:
            self.errors.append(f"sequence_length={seq_len} too short for meaningful patterns")
    
    def _validate_features(self, features: Dict[str, pd.DataFrame]):
        """Feature DataFrames validation"""
        expected_features = self.config['model']['d_input']
        
        for symbol, df in features.items():
            # Column count
            if df.shape[1] != expected_features:
                self.errors.append(
                    f"{symbol}: {df.shape[1]} features != expected {expected_features}"
                )
            
            # NaN check
            nan_pct = df.isna().sum().sum() / df.size * 100
            if nan_pct > 10:
                self.errors.append(f"{symbol}: {nan_pct:.1f}% NaN values")
            elif nan_pct > 1:
                self.warnings.append(f"{symbol}: {nan_pct:.1f}% NaN values")
            
            # Inf check
            numeric_df = df.select_dtypes(include=[np.number])
            if np.isinf(numeric_df.values).any():
                self.errors.append(f"{symbol}: contains inf values")
            
            # Length check
            seq_len = self.config['data']['sequence_length']
            if len(df) < seq_len:
                self.errors.append(
                    f"{symbol}: {len(df)} rows < sequence_length {seq_len}"
                )
    
    def _validate_labels(self, labels: Dict[str, pd.DataFrame]):
        """Label DataFrames validation"""
        for symbol, df in labels.items():
            # Must have at least 3 columns: direction, volatility, magnitude
            if df.shape[1] < 3:
                self.errors.append(
                    f"{symbol} labels: {df.shape[1]} columns, expected >= 3"
                )
                continue
            
            # Direction range [0, 2]
            direction = df.iloc[:, 0]
            if direction.min() < 0 or direction.max() > 2:
                self.errors.append(
                    f"{symbol} direction: [{direction.min()}, {direction.max()}], expected [0, 2]"
                )
            
            # Volatility range [0, 2]
            volatility = df.iloc[:, 1]
            if volatility.min() < 0 or volatility.max() > 2:
                self.errors.append(
                    f"{symbol} volatility: [{volatility.min()}, {volatility.max()}], expected [0, 2]"
                )
            
            # Magnitude range [0, 1]
            magnitude = df.iloc[:, 2]
            if magnitude.min() < 0 or magnitude.max() > 1:
                self.warnings.append(
                    f"{symbol} magnitude: [{magnitude.min():.2f}, {magnitude.max():.2f}], expected [0, 1]"
                )
    
    def _validate_alignment(self, features: Dict[str, pd.DataFrame], 
                           labels: Dict[str, pd.DataFrame]):
        """Features ↔ Labels alignment"""
        # Same symbols
        feat_symbols = set(features.keys())
        label_symbols = set(labels.keys())
        
        missing_labels = feat_symbols - label_symbols
        if missing_labels:
            self.errors.append(f"Symbols in features but not labels: {missing_labels}")
        
        missing_features = label_symbols - feat_symbols
        if missing_features:
            self.errors.append(f"Symbols in labels but not features: {missing_features}")
        
        # Same length per symbol
        for symbol in feat_symbols & label_symbols:
            feat_len = len(features[symbol])
            label_len = len(labels[symbol])
            
            if feat_len != label_len:
                self.errors.append(
                    f"{symbol}: features={feat_len}, labels={label_len} (must match)"
                )
    
    def _validate_model(self, model: torch.nn.Module):
        """Model architecture validation"""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Model: {total_params:,} params ({trainable_params:,} trainable)")
        
        # Estimate memory
        param_mb = total_params * 4 / 1024 / 1024  # float32
        print(f"   Estimated param memory: {param_mb:.1f} MB")
        
        # Check for NaN in weights (corrupted model)
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                self.errors.append(f"Model param {name} contains NaN")
            if torch.isinf(param).any():
                self.errors.append(f"Model param {name} contains inf")
    
    def _validate_dry_run(self, features: Dict[str, pd.DataFrame],
                          labels: Dict[str, pd.DataFrame],
                          model: torch.nn.Module):
        """Actually run one batch through"""
        try:
            if not features:
                raise ValueError("No features available for dry run")
                
            # Get first symbol
            symbol = list(features.keys())[0]
            feat_df = features[symbol]
            label_df = labels[symbol]
            
            # Create mini batch
            seq_len = min(self.config['data']['sequence_length'], len(feat_df))
            if seq_len == 0:
                 raise ValueError(f"Sequence length is 0 for {symbol}")
                 
            batch_size = 2
            
            # (B, L, F)
            X = torch.from_numpy(feat_df.iloc[:seq_len].values).float().unsqueeze(0)
            X = X.repeat(batch_size, 1, 1)
            
            # Handle Multi-Asset Input (B, N, L, F)
            n_assets = self.config['model'].get('n_assets', 1)
            if n_assets > 1:
                # Repeat for assets
                X = X.unsqueeze(1).repeat(1, n_assets, 1, 1)
            
            y = torch.from_numpy(label_df.iloc[:seq_len].values).unsqueeze(0)
            y = y.repeat(batch_size, 1, 1)  # (B, L, C)
            
            # Move to same device as model
            device = next(model.parameters()).device
            X = X.to(device)
            y = y.to(device)
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                outputs = model(X)
            
            # Validate outputs
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if key == 'fusion_weights':
                        continue
                    if key == 'multi_label':
                        if len(value) != 5:
                            self.errors.append(f"multi_label: {len(value)} outputs, expected 5")
                    elif isinstance(value, tuple) and len(value) == 3:
                        signal, conf, unc = value
                        if signal.shape[-1] != 3:
                            self.errors.append(
                                f"{key} signal: {signal.shape[-1]} classes, expected 3"
                            )
            
            print(f"   Dry run: X{tuple(X.shape)} -> outputs OK")
            
            # Test backward pass
            model.train()
            outputs = model(X)
            
            # Compute dummy loss
            if isinstance(outputs, dict):
                loss = torch.tensor(0.0, requires_grad=True, device=device)
                for key, value in outputs.items():
                    if key in ['fusion_weights', 'multi_label']:
                        continue
                    if isinstance(value, tuple):
                        signal, _, _ = value
                        loss = loss + signal.mean()
                
                loss.backward()
                print(f"   Backward pass: OK")
            
        except Exception as e:
            self.errors.append(f"Dry run failed: {str(e)}")
            import traceback
            traceback.print_exc()


def validate_pipeline(config: dict,
                      features: Dict[str, pd.DataFrame],
                      labels: Dict[str, pd.DataFrame],
                      model: torch.nn.Module) -> bool:
    """Convenience function"""
    validator = PipelineValidator(config)
    return validator.validate_all(features, labels, model)

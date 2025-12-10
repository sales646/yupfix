"""
Test entire pipeline with TINY data on CPU.
This validates the code works before deploying to GPU.
"""
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Force CPU
torch.set_default_device('cpu')

sys.path.insert(0, str(Path(__file__).parent.parent))

def create_tiny_data(n_rows: int = 1000):
    """Create minimal test data."""
    dates = pd.date_range('2024-01-01', periods=n_rows, freq='1s')
    
    price = 1.1000 + np.cumsum(np.random.randn(n_rows) * 0.0001)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price,
        'high': price + np.abs(np.random.randn(n_rows) * 0.0001),
        'low': price - np.abs(np.random.randn(n_rows) * 0.0001),
        'close': price + np.random.randn(n_rows) * 0.00005,
        'volume': np.random.randint(100, 1000, n_rows).astype(float),
        'buy_volume': np.random.randint(50, 500, n_rows).astype(float),
        'sell_volume': np.random.randint(50, 500, n_rows).astype(float),
        'bid_close': price - 0.00005,
        'ask_close': price + 0.00005,
        'spread_avg': np.full(n_rows, 0.0001),
        'spread_max': np.full(n_rows, 0.0002),
        'tick_count': np.random.randint(10, 50, n_rows),
    }, index=dates)
    
    return df


def test_pipeline():
    """Test complete pipeline on CPU with tiny data."""
    print("="*60)
    print("🧪 TINY CPU TEST")
    print("="*60)
    
    # 1. Test imports
    print("\n[1/7] Testing imports...")
    try:
        from src.yup250_pipeline import Yup250Pipeline
        from src.features.microstructure import MicrostructureFeatureEngineer
        from src.models.global_hmm import GlobalHMM
        from src.models.mamba_full import MambaTrader
        from src.targets.label_generator import create_trading_labels
        print("  ✅ All imports successful")
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False
    
    # 2. Create tiny data
    print("\n[2/7] Creating tiny test data...")
    data = {
        'EURUSD': create_tiny_data(1000),
        'GBPUSD': create_tiny_data(1000),
    }
    print(f"  ✅ Created data for {len(data)} symbols")
    
    # 3. Test feature engineering
    print("\n[3/7] Testing feature engineering...")
    try:
        engineer = MicrostructureFeatureEngineer(bar_seconds=1)
        features = {}
        for symbol, df in data.items():
            feat_list = engineer.compute_features(df)
            # Convert list of FeatureRow to DataFrame
            feat = pd.DataFrame([f.dict() for f in feat_list])
            if 'timestamp' in feat.columns:
                feat.set_index('timestamp', inplace=True)
            features[symbol] = feat
            print(f"  ✅ {symbol}: {feat.shape}")
    except Exception as e:
        print(f"  ❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Test label generation
    print("\n[4/7] Testing label generation...")
    try:
        labels = {}
        for symbol, df in data.items():
            lab = create_trading_labels(df, horizon=12, delay=3)
            labels[symbol] = lab
            print(f"  ✅ {symbol}: {lab.shape}")
    except Exception as e:
        print(f"  ❌ Label generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test HMM
    print("\n[5/7] Testing HMM...")
    try:
        hmm = GlobalHMM(n_states=3, window_seconds=60, bar_seconds=1)
        
        # Create returns
        returns = pd.DataFrame({
            f'asset_{i}': data[sym]['close'].pct_change()
            for i, sym in enumerate(data.keys())
        })
        
        hmm.fit(returns.dropna())
        states = hmm.predict(returns.dropna())
        print(f"  ✅ HMM fitted, states: {np.unique(states)}")
        print(f"  ℹ️ Using real HMM: {hmm.use_real_hmm}")
    except Exception as e:
        print(f"  ❌ HMM failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Test model creation
    print("\n[6/7] Testing model creation...")
    try:
        # TINY config for CPU testing
        model = MambaTrader(
            d_input=21,      # Match microstructure features
            d_model=32,      # Small for CPU
            n_layers=2,      # Few layers
            d_state=16,      # Small state
            n_assets=2,
            use_multi_label=True
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Model created: {total_params:,} parameters")
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. Test forward pass
    print("\n[7/7] Testing forward pass...")
    try:
        # Get feature count
        n_features = features['EURUSD'].shape[1]
        seq_len = min(100, len(features['EURUSD']))
        
        # Create batch (Batch, N_Assets, SeqLen, Features)
        # n_assets=2 in init
        X = torch.randn(2, 2, seq_len, n_features)  # (2, 2, 100, 21)
        
        model.eval()
        with torch.no_grad():
            outputs = model(X)
        
        print(f"  ✅ Forward pass successful")
        print(f"     Input: {tuple(X.shape)}")
        print(f"     Outputs: {list(outputs.keys())}")
        
        # Test backward
        model.train()
        outputs = model(X)
        loss = sum(v[0].mean() for k, v in outputs.items() 
                   if k not in ['fusion_weights', 'multi_label'] and isinstance(v, tuple))
        loss.backward()
        print(f"  ✅ Backward pass successful")
        
    except Exception as e:
        print(f"  ❌ Forward/backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nCode is ready for GPU deployment.")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)

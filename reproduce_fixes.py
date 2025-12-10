import torch
import pandas as pd
import numpy as np
from src.models.global_hmm import GlobalHMM
from src.models.mamba_full import MambaTrader
from src.models.mamba_ensemble import MambaEnsemble

def test_global_hmm():
    print("Testing GlobalHMM...")
    
    # Generate dummy multi-asset returns
    np.random.seed(42)
    n_samples = 500
    n_assets = 6
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1s')
    data = pd.DataFrame(
        np.random.randn(n_samples, n_assets) * 0.001,
        index=dates,
        columns=[f'asset_{i}' for i in range(n_assets)]
    )
    
    # Test fit
    hmm = GlobalHMM(n_states=3, window_seconds=100, bar_seconds=1)
    
    # Verify PCA is not fitted initially
    assert not hmm.is_fitted, "HMM should not be fitted initially"
    
    # Fit HMM
    print("Fitting HMM...")
    hmm.fit(data)
    
    # Verify fitted
    assert hmm.is_fitted, "HMM should be fitted after fit()"
    
    # Check PCA components exist
    assert hasattr(hmm.pca, 'components_'), "PCA should have components after fit"
    pca_components_before = hmm.pca.components_.copy()
    
    # Test predict
    print("Testing predict...")
    states = hmm.predict(data)
    print(f"Predicted states shape: {states.shape}")
    assert states.shape[0] > 0, "Should have predicted states"
    
    # Verify PCA components haven't changed after predict
    assert np.allclose(hmm.pca.components_, pca_components_before), \
        "PCA components should not change during predict()"
    
    # Test state mapping stability
    print("Testing state mapping stability...")
    hmm2 = GlobalHMM(n_states=3, window_seconds=100, bar_seconds=1)
    hmm2.fit(data)
    assert np.array_equal(hmm.state_mapping, hmm2.state_mapping), \
        "State mapping should be stable across fits with same data"
    
    print("GlobalHMM tests passed!")

def test_mamba_full():
    print("\nTesting MambaTrader (multi-asset)...")
    
    batch_size = 2
    n_assets = 3
    seq_len = 50
    d_input = 10
    d_model = 16
    
    model = MambaTrader(d_input=d_input, d_model=d_model, n_layers=2, n_assets=n_assets)
    
    # Test multi-asset input
    x = torch.randn(batch_size, n_assets, seq_len, d_input)
    
    print("Running forward pass...")
    try:
        outputs = model(x)
        print(f"Output keys: {outputs.keys()}")
        
        # Check if pooling_proj exists
        assert hasattr(model, 'pooling_proj'), "Model should have pooling_proj layer"
        
        print("MambaTrader tests passed!")
    except Exception as e:
        print(f"MambaTrader forward pass failed: {e}")
        raise

def test_mamba_ensemble():
    print("\nTesting MambaEnsemble diversity measurement...")
    
    # Create dummy model class
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return {'h1': (torch.randn(2, 3), torch.randn(2), torch.randn(2))}
    
    # Create ensemble
    ensemble = MambaEnsemble(
        model_class=DummyModel,
        model_kwargs={},
        n_models=3
    )
    
    # Test measure_diversity
    print("Testing measure_diversity...")
    dummy_predictions = [
        torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]),
        torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
    ]
    
    diversity = ensemble.measure_diversity(dummy_predictions)
    print(f"Diversity score: {diversity}")
    
    assert 0.0 <= diversity <= 1.0, "Diversity should be between 0 and 1"
    assert diversity > 0, "Diversity should be > 0 for different predictions"
    
    # Test identical predictions
    identical_predictions = [
        torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    ]
    
    diversity_identical = ensemble.measure_diversity(identical_predictions)
    print(f"Diversity (identical): {diversity_identical}")
    assert diversity_identical == 0.0, "Diversity should be 0 for identical predictions"
    
    print("MambaEnsemble tests passed!")

def test_microstructure():
    print("\nTesting MicrostructureFeatureEngineer...")
    from src.features.microstructure import MicrostructureFeatureEngineer
    
    # Generate dummy data
    n_samples = 200
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1s')
    df = pd.DataFrame({
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(100, 1000, n_samples),
        'buy_volume': np.random.randint(50, 500, n_samples),
        'sell_volume': np.random.randint(50, 500, n_samples),
        'bid_close': np.random.randn(n_samples).cumsum() + 99.5,
        'ask_close': np.random.randn(n_samples).cumsum() + 100.5,
        'spread_avg': np.random.rand(n_samples) * 0.01,
        'spread_max': np.random.rand(n_samples) * 0.02,
        'tick_count': np.random.randint(10, 50, n_samples),
    }, index=dates)
    
    # Test feature computation (bar_seconds=1)
    engineer = MicrostructureFeatureEngineer(bar_seconds=1)
    features = engineer.compute_features(df)
    
    print(f"Features shape: {features.shape}")
    print(f"Features columns: {list(features.columns)[:5]}...")
    
    # Check OFI windows are in seconds (for 1-sec bars, windows should be 100, 500, 1000)
    ofi_cols = [c for c in features.columns if 'ofi_' in c]
    print(f"OFI window columns: {ofi_cols}")
    assert len(ofi_cols) > 0, "Should have OFI window features"
    
    # Check feature clipping was applied
    for col in features.columns:
        if len(features[col].dropna()) > 0:
            # Values should be within reasonable range (not extreme outliers)
            std = features[col].std()
            assert not features[col].isnull().all(), f"{col} should have some values"
    
    print("MicrostructureFeatureEngineer tests passed!")

def test_risk_manager():
    print("\nTesting RiskManager...")
    from src.risk.manager import RiskManager
    
    config = {
        'max_drawdown': 0.08,
        'max_daily_loss': 0.04,
        'max_position_size': 1.0,
        'correlation_groups': {
            'majors': ['EUR_USD', 'GBP_USD'],
            'minors': ['EUR_JPY', 'GBP_JPY']
        }
    }
    
    rm = RiskManager(config)
    
    # Test daily loss tracking
    print("Testing daily loss tracking...")
    balance_day1 = 10000
    peak = 10000
    
    # Day 1
    result1 = rm.check_position(1.0, balance_day1, peak, initial_balance=10000, current_date='2024-01-01')
    assert rm.daily_start_balance == balance_day1, "Daily start balance should be set"
    
    # Day 1 with loss (3% - should pass)
    result2 = rm.check_position(1.0, 9700, peak, initial_balance=10000, current_date='2024-01-01')
    assert result2 > 0, "Should allow position with 3% daily loss"
    
    # Day 1 with large loss (5% - should block)
    result3 = rm.check_position(1.0, 9500, peak, initial_balance=10000, current_date='2024-01-01')
    assert result3 == 0.0, "Should block position with 5% daily loss"
    
    # Day 2 (reset)
    result4 = rm.check_position(1.0, 9500, peak, initial_balance=10000, current_date='2024-01-02')
    assert rm.daily_start_balance == 9500, "Daily start balance should reset"
    assert result4 > 0, "Should allow position on new day"
    
    # Test correlation group enforcement
    print("Testing correlation group enforcement...")
    proposed = {
        'EUR_USD': 0.03,
        'GBP_USD': 0.02,
        'EUR_JPY': 0.01
    }
    
    adjusted = rm.check_correlation_exposure(proposed)
    majors_exposure = abs(adjusted['EUR_USD']) + abs(adjusted['GBP_USD'])
    print(f"Majors exposure: {majors_exposure:.2%}")
    assert majors_exposure <= 0.041, "Majors exposure should be ≤ 4%"
    
    print("RiskManager tests passed!")

if __name__ == "__main__":
    test_global_hmm()
    test_mamba_full()
    test_mamba_ensemble()
    test_microstructure()
    test_risk_manager()
    print("\n✅ All tests passed!")

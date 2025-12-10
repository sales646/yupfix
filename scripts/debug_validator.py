import torch
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.contracts.pipeline_validator import validate_pipeline
from src.models.mamba_full import MambaTrader

def debug_validator():
    print("DEBUG: Starting validator debug")
    
    # Dummy config
    config = {
        'data': {
            'symbols': ['EURUSD', 'XAUUSD'],
            'sequence_length': 100,
        },
        'model': {
            'd_input': 24,
            'n_assets': 2,
            'd_model': 64,
            'n_layers': 2,
            'd_state': 16,
        }
    }
    
    # Dummy features
    features = {
        'EURUSD': pd.DataFrame(np.random.randn(200, 24)),
        'XAUUSD': pd.DataFrame(np.random.randn(200, 24))
    }
    
    # Dummy labels
    labels = {
        'EURUSD': pd.DataFrame(np.random.randint(0, 3, (200, 3))),
        'XAUUSD': pd.DataFrame(np.random.randint(0, 3, (200, 3)))
    }
    
    # Dummy model
    model = MambaTrader(
        d_input=24,
        d_model=64,
        n_layers=2,
        d_state=16,
        n_assets=2
    )
    
    print("DEBUG: Calling validate_pipeline")
    try:
        validate_pipeline(config, features, labels, model)
        print("DEBUG: Validation successful")
    except Exception as e:
        print(f"DEBUG: Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_validator()

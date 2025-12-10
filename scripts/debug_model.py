import sys
from pathlib import Path
import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mamba_full import MambaTrader

def debug_model():
    print("Loading config...")
    with open("config/training.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    model_config = config['model']
    # Remove dropout if present (as we did in pipeline)
    if 'dropout' in model_config:
        del model_config['dropout']
        
    print(f"Model config: {model_config}")
    
    print("Creating model...")
    try:
        model = MambaTrader(**model_config)
        print("Model created successfully.")
    except Exception as e:
        print(f"Failed to create model: {e}")
        return

    # Create dummy input
    batch_size = 2
    seq_len = 64
    d_input = model_config['d_input']
    
    x = torch.randn(batch_size, seq_len, d_input)
    print(f"Input shape: {x.shape}")
    
    print("Running forward pass...")
    try:
        outputs = model(x)
        print("Forward pass successful.")
        print(f"Output type: {type(outputs)}")
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}")
                elif isinstance(v, (list, tuple)):
                    print(f"  {k}: len={len(v)}")
                    for i, item in enumerate(v):
                        if isinstance(item, torch.Tensor):
                            print(f"    [{i}]: {item.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model()

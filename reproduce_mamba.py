import torch
import time
from src.models.mamba_block import MambaBlock
from src.models.mamba_backbone import MambaBackbone

def test_mamba_block():
    print("Testing MambaBlock...")
    batch_size = 2
    seq_len = 100
    d_model = 16
    
    block = MambaBlock(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test forward pass
    try:
        y = block(x)
        print(f"Forward pass successful. Output shape: {y.shape}")
        assert y.shape == (batch_size, seq_len, d_model)
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise

def test_mamba_backbone():
    print("\nTesting MambaBackbone...")
    batch_size = 2
    seq_len = 100
    d_input = 10
    d_model = 16
    n_layers = 2
    
    backbone = MambaBackbone(d_input=d_input, d_model=d_model, n_layers=n_layers)
    x = torch.randn(batch_size, seq_len, d_input)
    timestamps = torch.rand(batch_size, seq_len, 1)
    
    # Test forward pass with timestamps and checkpointing
    try:
        y = backbone(x, timestamps=timestamps, use_checkpoint=True)
        print(f"Forward pass successful. Output shape: {y.shape}")
        assert y.shape == (batch_size, seq_len, d_model)
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise
        
    # Test backward pass (to verify checkpointing)
    try:
        loss = y.sum()
        loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"Backward pass failed: {e}")
        raise

if __name__ == "__main__":
    test_mamba_block()
    test_mamba_backbone()
    print("\nAll tests passed!")

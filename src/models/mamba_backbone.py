"""
Mamba-2 Backbone: Stacked Mamba blocks
"""
import torch
import torch.nn as nn
from .mamba_block import MambaBlock

class MambaBackbone(nn.Module):
    """
    Stacked Mamba blocks with residual connections and layer norm.
    
    Config:
    - d_model: 256
    - n_layers: 12
    - d_state: 128
    - sequence_length: 14400 (4 hours @ 1-sec)
    """
    def __init__(self, d_input: int, d_model: int = 256, n_layers: int = 12,
                 d_state: int = 128, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Input embedding
        self.input_proj = nn.Linear(d_input, d_model)
        
        # Time embedding
        self.time_embed = nn.Linear(1, d_model)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Layer norms (pre-norm architecture)
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, timestamps=None, use_checkpoint: bool = True):
        """
        Args:
            x: (Batch, SeqLen, d_input)
            timestamps: (Batch, SeqLen, 1) normalized 0-1, optional
            use_checkpoint: bool, whether to use gradient checkpointing
        Returns:
            (Batch, SeqLen, d_model)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Add time embedding if provided
        if timestamps is not None:
            time_emb = self.time_embed(timestamps)
            x = x + time_emb
        
        # Mamba layers with residual
        from torch.utils.checkpoint import checkpoint
        
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            if use_checkpoint and self.training:
                # Checkpoint every 2nd layer to save memory
                if i % 2 == 0:
                    # checkpoint requires the function to have requires_grad=True inputs
                    # layer(norm(x)) is the operation
                    x = x + checkpoint(lambda y: layer(norm(y)), x, use_reentrant=False)
                else:
                    x = x + layer(norm(x))
            else:
                x = x + layer(norm(x))
        
        return self.final_norm(x)

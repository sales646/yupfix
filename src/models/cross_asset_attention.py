"""
Cross-Asset Attention Module
Enables information flow between different assets (e.g., EURUSD, NAS100).
"""
import torch
import torch.nn as nn

class CrossAssetAttention(nn.Module):
    """
    Deep Cross-Asset Attention with Asset Embeddings.
    Uses a Transformer Encoder to mix information across assets.
    """
    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 3, n_assets: int = 6, dropout: float = 0.1):
        super().__init__()
        
        # Asset Embeddings: Give each asset a unique identity
        # This helps the model distinguish between EURUSD and NAS100 even if features look similar
        self.asset_embed = nn.Embedding(n_assets, d_model)
        
        # Deep Transformer Encoder
        # batch_first=True expects (Batch, Seq, Feature) -> (Batch, N_Assets, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True # Pre-norm is generally more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x):
        """
        Args:
            x: (Batch, N_Assets, d_model)
            
        Returns:
            (Batch, N_Assets, d_model)
        """
        batch_size, n_assets, d_model = x.shape
        
        # Create asset IDs: [0, 1, ..., N-1] repeated for batch
        device = x.device
        asset_ids = torch.arange(n_assets, device=device).unsqueeze(0).expand(batch_size, -1) # (B, N)
        
        # Add asset embeddings to input features
        # Broadcasting: (B, N, D) + (B, N, D)
        x = x + self.asset_embed(asset_ids)
        
        # Pass through Transformer
        # Attention happens across the N_Assets dimension
        x = self.transformer(x)
        
        return x

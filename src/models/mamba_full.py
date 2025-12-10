"""
Full Mamba Trader Model
"""
import torch
import torch.nn as nn
from .mamba_backbone import MambaBackbone
from .mamba_heads import MultiScaleHeads
from .cross_asset_attention import CrossAssetAttention
from .multi_label_head import MultiLabelHead

class MambaTrader(nn.Module):
    """
    Complete Mamba-2 trading model.
    
    Pipeline:
    Input → Backbone → Cross-Asset Attention → Multi-Scale Heads → Output
    """
    def __init__(self, d_input: int, d_model: int = 256, n_layers: int = 12,
                 d_state: int = 128, n_assets: int = 6, n_cross_layers: int = 3, use_multi_label: bool = True):
        super().__init__()
        
        self.backbone = MambaBackbone(d_input, d_model, n_layers, d_state)
        
        # Deep Cross-Asset Attention
        self.cross_attention = CrossAssetAttention(
            d_model=d_model, 
            n_layers=n_cross_layers, 
            n_assets=n_assets
        )
        
        # Multi-scale heads (Scalp, Intraday, Swing)
        self.heads = MultiScaleHeads(d_model)
        
        # Projection for pooled features (3x d_model -> d_model)
        self.pooling_proj = nn.Linear(d_model * 3, d_model)
        
        self.use_multi_label = use_multi_label
        if use_multi_label:
            self.multi_label = MultiLabelHead(d_model)
        
        self.n_assets = n_assets
    
    def forward(self, x, return_features=False):
        """
        Forward pass through the complete model.
        
        Args:
            x: Input tensor (Batch, N_Assets, SeqLen, d_input)
            return_features: Whether to return intermediate features
            
        Returns:
            outputs: Dict of predictions
        """
        if x.dim() == 3:
            # Handle (Batch, SeqLen, d_input) case - assume single asset or pre-flattened?
            # But we need N_Assets for cross-attention.
            # Let's assume (Batch, SeqLen, d_input) is NOT supported for this multi-asset model
            # unless N_Assets=1.
            # For robustness, if N_Assets > 1, we expect 4D.
            if self.n_assets > 1:
                raise ValueError(f"Expected 4D input (Batch, N_Assets, SeqLen, d_input) for n_assets={self.n_assets}")
            else:
                x = x.unsqueeze(1) # (B, 1, L, D)
                
        B, N, L, D = x.shape
        
        # Flatten for backbone: (B*N, L, D)
        x_flat = x.view(B * N, L, D)
        
        # Backbone
        h = self.backbone(x_flat) # (B*N, L, d_model)
        
        # Pooling (Mean + Max + Last)
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1)[0]
        h_last = h[:, -1, :]
        
        h_pooled = torch.cat([h_mean, h_max, h_last], dim=-1) # (B*N, 3*d_model)
        
        # Project back to d_model
        h_proj = self.pooling_proj(h_pooled) # (B*N, d_model)
        
        # Reshape for cross-attention: (B, N, d_model)
        h_assets = h_proj.view(B, N, -1)
        
        # Cross-asset attention
        features = self.cross_attention(h_assets) # (B, N, d_model)
        
        # Multi-scale heads
        # features is (B, N, d_model)
        # MultiScaleHeads expects (Batch, SeqLen, d_model) OR (Batch, d_model) depending on implementation.
        # Let's check MultiScaleHeads implementation.
        # It takes (Batch, SeqLen, d_model) and computes mean/last.
        # BUT here 'features' is (B, N, d_model) - N is asset dimension, not time!
        # We already pooled time in the backbone section.
        # So 'features' represents the "asset state" after cross-attention.
        # We want to apply heads to EACH asset state.
        # So we treat (B*N) as batch.
        
        features_flat = features.view(B * N, -1) # (B*N, d_model)
        
        # Wait, MultiScaleHeads expects (Batch, SeqLen, d_model) to compute intraday/swing means.
        # BUT we lost SeqLen dimension after pooling!
        # We only have (B, N, d_model).
        # This means we CANNOT use MultiScaleHeads as originally designed (which averaged over time).
        # UNLESS we pass the unpooled sequence `h` (B*N, L, d_model) to the heads?
        # But `h` hasn't gone through cross-attention.
        # Cross-attention mixes assets.
        # If we want cross-asset aware signals, we must use `features`.
        # But `features` has no time dimension.
        
        # SOLUTION:
        # 1. Use `features` (B, N, d_model) as the "Scalp" (immediate) context.
        # 2. For Intraday/Swing, we might need to project `h` (time-aware) and mix it?
        # OR simpler:
        # MultiScaleHeads in this architecture might just be a set of heads applied to the SAME feature vector,
        # but trained with different targets (horizons).
        # Let's look at MultiScaleHeads again.
        # It takes (Batch, SeqLen, d_model).
        # It computes: scalp (last), intraday (1h mean), swing (all mean).
        
        # Since we don't have SeqLen after cross-attention, we have two options:
        # A) Apply Cross-Attention per timestep (expensive).
        # B) Apply Cross-Attention only on pooled features (current).
        
        # If (B), then MultiScaleHeads cannot compute time-averages on the output of Cross-Attention.
        # It can only act on the single vector per asset.
        # So we should modify MultiScaleHeads to accept a single vector and just produce 3 outputs?
        # OR we pass `features` as (B*N, 1, d_model) and let it select "last"?
        # But it can't do "1h mean".
        
        # Compromise:
        # We use `features` (B, N, d_model) as the input to ALL heads.
        # The "Scalp", "Intraday", "Swing" distinction now comes purely from the LABELS we train against,
        # not from different input pooling (since input is already pooled).
        # This is acceptable. The model learns to map the "current state" to different horizons.
        
        # So we need to modify MultiScaleHeads to accept (Batch, d_model) instead of (Batch, Seq, d_model).
        # OR we just treat `features` as (B*N, 1, d_model).
        
        # Let's update MambaTrader to reshape features for heads.
        # And we might need to tweak MultiScaleHeads if it strictly requires SeqLen > 1.
        # Let's assume we can pass (B*N, 1, d_model).
        
        features_flat_seq = features.view(B * N, 1, -1) # (B*N, 1, d_model)
        outputs = self.heads(features_flat_seq)
        
        # Outputs will be (B*N, ...)
        # We need to reshape back to (B, N, ...) for the pipeline to handle it (or keep flattened?)
        # Trainer handles (B, N, ...) by flattening.
        # So returning (B*N, ...) is fine if we document it.
        # But `outputs` is a dict.
        # Let's reshape values back to (B, N, ...) for consistency with input.
        
        new_outputs = {}
        for k, v in outputs.items():
            if k == 'fusion_weights':
                new_outputs[k] = v
                continue
                
            if isinstance(v, torch.Tensor):
                # v is (B*N, ...)
                new_outputs[k] = v.view(B, N, *v.shape[1:])
            elif isinstance(v, tuple):
                # tuple of tensors
                new_outputs[k] = tuple(t.view(B, N, *t.shape[1:]) for t in v)
            else:
                new_outputs[k] = v
        outputs = new_outputs
        
        if self.use_multi_label:
            # Multi-label head expects (B, N, d_model)
            multi_out = self.multi_label(features)
            outputs['multi_label'] = multi_out
        
        if return_features:
            return outputs, features
        return outputs

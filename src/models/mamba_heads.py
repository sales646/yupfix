"""
Multi-scale prediction heads for Mamba
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictionHead(nn.Module):
    """
    Single prediction head with uncertainty estimation.
    
    Outputs:
    - signal: Softmax over [-1, 0, +1] (SHORT/FLAT/LONG)
    - confidence: Sigmoid [0, 1]
    - uncertainty: Softplus [0, ∞)
    """
    def __init__(self, d_model: int, n_classes: int = 3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        self.signal_head = nn.Linear(d_model // 2, n_classes)
        self.confidence_head = nn.Linear(d_model // 2, 1)
        self.uncertainty_head = nn.Linear(d_model // 2, 1)
        
    def forward(self, x):
        """
        Args:
            x: (Batch, d_model)
        Returns:
            signal: (Batch, n_classes) logits
            confidence: (Batch, 1)
            uncertainty: (Batch, 1)
        """
        features = self.proj(x)
        signal = self.signal_head(features)
        confidence = torch.sigmoid(self.confidence_head(features))
        uncertainty = F.softplus(self.uncertainty_head(features))
        return signal, confidence, uncertainty


class MultiScaleHeads(nn.Module):
    """
    Multiple prediction heads for different time horizons.
    
    Horizons:
    - Scalp: Last token (immediate)
    - Intraday: 1-hour average
    - Swing: Full sequence average (4 hours)
    """
    def __init__(self, d_model: int, intraday_window: int = 3600):
        super().__init__()
        self.intraday_window = intraday_window
        
        self.scalp_head = PredictionHead(d_model)
        self.intraday_head = PredictionHead(d_model)
        self.swing_head = PredictionHead(d_model)
        
        # Fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        """
        Args:
            x: (Batch, SeqLen, d_model) from Mamba backbone
        Returns:
            Dict with predictions per horizon
        """
        batch_size, seq_len, d_model = x.shape
        
        # Different temporal contexts
        scalp_features = x[:, -1, :]  # Last token
        intraday_features = x[:, -self.intraday_window:, :].mean(dim=1)  # 1h avg
        swing_features = x.mean(dim=1)  # Full avg
        
        # Get predictions
        scalp_out = self.scalp_head(scalp_features)
        intraday_out = self.intraday_head(intraday_features)
        swing_out = self.swing_head(swing_features)
        
        return {
            'scalp': scalp_out,
            'intraday': intraday_out,
            'swing': swing_out,
            'fusion_weights': F.softmax(self.fusion_weights, dim=0)
        }

"""
Multi-Label Prediction Head for comprehensive trading signals
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelHead(nn.Module):
    """
    Multi-Label Prediction Head.
    
    Outputs:
    1. Direction (Buy/Hold/Sell) - 3 classes
    2. Volatility (Low/Medium/High) - 3 classes  
    3. Magnitude (Continuous, 0-1 normalized)
    4. Confidence (0-1)
    5. Uncertainty (0+)
    """
    def __init__(self, d_model: int):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        self.direction_head = nn.Linear(d_model // 2, 3)
        self.volatility_head = nn.Linear(d_model // 2, 3)
        self.magnitude_head = nn.Linear(d_model // 2, 1)
        self.confidence_head = nn.Linear(d_model // 2, 1)
        self.uncertainty_head = nn.Linear(d_model // 2, 1)

    def forward(self, x: torch.Tensor):
        features = self.proj(x)
        
        direction = self.direction_head(features)
        volatility = self.volatility_head(features)
        magnitude = torch.sigmoid(self.magnitude_head(features))
        confidence = torch.sigmoid(self.confidence_head(features))
        uncertainty = F.softplus(self.uncertainty_head(features))
        
        return direction, volatility, magnitude, confidence, uncertainty
    
    def predict(self, x: torch.Tensor):
        """Inference-friendly output"""
        direction, volatility, magnitude, confidence, uncertainty = self.forward(x)
        
        direction_probs = F.softmax(direction, dim=-1)
        volatility_probs = F.softmax(volatility, dim=-1)
        direction_class = direction_probs.argmax(dim=-1) - 1  # -1, 0, +1
        volatility_class = volatility_probs.argmax(dim=-1)    # 0, 1, 2
        
        return {
            'signal': direction_class,
            'direction_probs': direction_probs,
            'volatility_class': volatility_class,
            'volatility_probs': volatility_probs,
            'magnitude': magnitude,
            'confidence': confidence,
            'uncertainty': uncertainty,
        }


class MultiTaskLoss(nn.Module):
    """Multi-Task Loss combining all outputs."""
    def __init__(self, 
                 direction_weight: float = 1.0,
                 volatility_weight: float = 0.5,
                 magnitude_weight: float = 0.3,
                 confidence_weight: float = 0.2,
                 uncertainty_weight: float = 0.1):
        super().__init__()
        self.weights = {
            'direction': direction_weight,
            'volatility': volatility_weight,
            'magnitude': magnitude_weight,
            'confidence': confidence_weight,
            'uncertainty': uncertainty_weight
        }
        
    def forward(self, predictions, targets):
        losses = {}
        losses['direction'] = F.cross_entropy(predictions[0], targets['direction'])
        losses['volatility'] = F.cross_entropy(predictions[1], targets['volatility'])
        losses['magnitude'] = F.mse_loss(predictions[2], targets['magnitude'])
        losses['confidence'] = F.binary_cross_entropy(
            predictions[3].squeeze(-1), targets['confidence']
        )
        losses['uncertainty'] = F.mse_loss(predictions[4], targets['uncertainty'])
        
        total = sum(self.weights[k] * losses[k] for k in losses)
        
        return total, {k: v.item() for k, v in losses.items()}

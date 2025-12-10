"""
Drawdown-Aware Loss Functions
Adds risk-aware penalties to make the model learn to avoid large drawdowns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DrawdownAwareLoss(nn.Module):
    """
    Loss function that penalizes predictions leading to large drawdowns.
    
    The model learns not just to predict direction, but to:
    1. Avoid predictions that lead to large losses
    2. Reduce confidence during high-volatility periods
    3. Prefer consistent small gains over volatile returns
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Risk thresholds (matching FTMO limits)
        self.daily_limit = config.get('daily_loss_limit', 0.05)    # 5% daily
        self.max_limit = config.get('max_loss_limit', 0.10)        # 10% total
        
        # Penalty weights
        self.drawdown_penalty_weight = config.get('drawdown_penalty', 2.0)
        self.volatility_penalty_weight = config.get('volatility_penalty', 0.5)
        self.consistency_bonus = config.get('consistency_bonus', 0.3)
        
        # Tracking
        self.cumulative_pnl = 0.0
        self.peak_pnl = 0.0
        self.daily_pnl = 0.0
        
    def reset_tracking(self):
        """Reset at start of each epoch."""
        self.cumulative_pnl = 0.0
        self.peak_pnl = 0.0
        self.daily_pnl = 0.0
        
    def compute_simulated_returns(self, 
                                   predictions: torch.Tensor, 
                                   targets: torch.Tensor,
                                   confidence: torch.Tensor = None) -> torch.Tensor:
        """
        Simulate returns based on predictions and actual outcomes.
        
        Args:
            predictions: (B, 3) class probabilities [down, neutral, up]
            targets: (B,) actual direction class
            confidence: (B,) model confidence
            
        Returns:
            (B,) simulated per-sample returns
        """
        B = predictions.shape[0]
        
        # Get predicted direction
        pred_direction = predictions.argmax(dim=-1)  # 0=down, 1=neutral, 2=up
        
        # Map to position: down=-1, neutral=0, up=1
        pred_position = pred_direction.float() - 1.0  # [-1, 0, 1]
        
        # Map target to actual move: down=-1, neutral=0, up=1
        actual_move = targets.float() - 1.0  # [-1, 0, 1]
        
        # Return = position * actual_move
        # Correct prediction: +1 (or fraction based on position)
        # Wrong prediction: -1 (or fraction)
        raw_return = pred_position * actual_move
        
        # Scale by confidence if available
        if confidence is not None:
            # Higher confidence = bigger position
            return raw_return * confidence.squeeze()
        
        return raw_return
    
    def compute_drawdown_penalty(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty based on drawdown from peak.
        
        Args:
            returns: (B,) returns per sample
            
        Returns:
            Scalar penalty tensor
        """
        # Cumulative returns in batch
        cumsum = returns.cumsum(dim=0)
        
        # Running peak
        running_peak = cumsum.cummax(dim=0).values
        
        # Drawdown = peak - current
        drawdowns = running_peak - cumsum
        
        # Penalize drawdowns that exceed thresholds
        # Soft penalty: increases exponentially as DD approaches limit
        daily_excess = F.relu(drawdowns - self.daily_limit * 0.5)  # Start penalizing at 2.5%
        max_excess = F.relu(drawdowns - self.max_limit * 0.5)      # Start penalizing at 5%
        
        # Exponential penalty
        daily_penalty = (torch.exp(daily_excess * 10) - 1).mean()
        max_penalty = (torch.exp(max_excess * 5) - 1).mean()
        
        return daily_penalty + max_penalty * 2.0
    
    def compute_volatility_penalty(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Penalize high volatility of returns.
        Encourages consistent small gains over volatile big swings.
        """
        if returns.numel() < 2:
            return torch.tensor(0.0, device=returns.device)
            
        return returns.std() * self.volatility_penalty_weight
    
    def compute_consistency_reward(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Reward consistent positive returns.
        """
        # Count positive returns
        positive_ratio = (returns > 0).float().mean()
        
        # Reward if >50% positive
        if positive_ratio > 0.5:
            return -self.consistency_bonus * (positive_ratio - 0.5)
        return torch.tensor(0.0, device=returns.device)
    
    def forward(self, 
                outputs: Dict,
                targets: torch.Tensor,
                base_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss with risk penalties.
        
        Args:
            outputs: Model outputs dict
            targets: Target labels
            base_loss: Base prediction loss
            
        Returns:
            (total_loss, loss_breakdown)
        """
        loss_breakdown = {'base_loss': base_loss.item()}
        
        # Extract predictions and confidence from outputs
        predictions = None
        confidence = None
        
        if isinstance(outputs, dict):
            # Get primary horizon (scalp or first available)
            for key in ['scalp', 'intraday', 'swing']:
                if key in outputs:
                    signal, conf, unc = outputs[key]
                    predictions = signal
                    confidence = conf
                    break
        
        if predictions is None:
            # No risk penalty if can't extract predictions
            return base_loss, loss_breakdown
        
        # Get direction target
        if targets.dim() == 3:
            direction_target = targets[:, -1, 0].long()
        else:
            direction_target = targets[:, -1].long() if targets.dim() == 2 else targets.long()
        
        # Simulate returns
        returns = self.compute_simulated_returns(predictions, direction_target, confidence)
        
        # Compute penalties
        dd_penalty = self.compute_drawdown_penalty(returns)
        vol_penalty = self.compute_volatility_penalty(returns)
        consistency = self.compute_consistency_reward(returns)
        
        # Total loss
        total_loss = (
            base_loss + 
            self.drawdown_penalty_weight * dd_penalty +
            vol_penalty +
            consistency
        )
        
        # Log breakdown
        loss_breakdown.update({
            'drawdown_penalty': dd_penalty.item() if isinstance(dd_penalty, torch.Tensor) else dd_penalty,
            'volatility_penalty': vol_penalty.item() if isinstance(vol_penalty, torch.Tensor) else vol_penalty,
            'consistency_reward': consistency.item() if isinstance(consistency, torch.Tensor) else consistency,
            'total_loss': total_loss.item()
        })
        
        return total_loss, loss_breakdown


def compute_loss_with_risk(outputs, y, config, risk_loss_fn=None):
    """
    Wrapper that adds risk-aware penalties to base loss.
    Drop-in replacement for original compute_loss.
    """
    from src.training.trainer import compute_loss
    
    # Compute base loss
    base_loss = compute_loss(outputs, y, config)
    
    # Add risk penalties if enabled
    if risk_loss_fn is not None:
        total_loss, breakdown = risk_loss_fn(outputs, y, base_loss)
        return total_loss, breakdown
    
    return base_loss, {'base_loss': base_loss.item(), 'total_loss': base_loss.item()}

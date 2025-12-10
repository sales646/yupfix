"""
Regime-Specific Mamba Model
Ensemble of Mamba models specialized for different market regimes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_full import MambaTrader
from .global_hmm import GlobalHMM

class RegimeSpecificMamba(nn.Module):
    """
    Mixture of Experts where experts are regime-specific Mamba models.
    
    Structure:
    - Regime 0: Stable/Low Volatility
    - Regime 1: Uncertain/Transition
    - Regime 2: Crisis/High Volatility
    
    Routing is determined by HMM probabilities or a learned gate.
    """
    def __init__(self, mamba_config: dict, n_regimes: int = 3, freeze_hmm: bool = True):
        super().__init__()
        
        self.n_regimes = n_regimes
        
        # Initialize separate Mamba models for each regime
        self.experts = nn.ModuleList([
            MambaTrader(**mamba_config) for _ in range(n_regimes)
        ])
        
        # Gating mechanism
        # If we use HMM probs directly, we don't need a learned gate, 
        # but a learned gate can fine-tune the HMM signal.
        # Input: Regime probabilities (3) -> Output: Mixing weights (3)
        self.gate = nn.Sequential(
            nn.Linear(n_regimes, 16),
            nn.ReLU(),
            nn.Linear(16, n_regimes)
        )
        
    def forward(self, x, regime_probs):
        """
        Args:
            x: Input features (B, L, D)
            regime_probs: HMM probabilities for the current state (B, n_regimes)
        """
        # Get predictions from each expert
        expert_outputs = []
        for expert in self.experts:
            out = expert(x)
            # Assuming output is a dict, we need to handle that.
            # For simplicity, let's assume we blend the 'direction' logits.
            # If we want to blend everything, it gets complex.
            # Strategy: Blend the FINAL LOGITS.
            expert_outputs.append(out)
            
        # Compute mixing weights from regime probs
        # We can use regime_probs directly or pass through gate
        gate_logits = self.gate(regime_probs)
        weights = F.softmax(gate_logits, dim=-1) # (B, n_regimes)
        
        # Blend outputs
        # We need to blend each key in the output dict
        blended_output = {}
        
        # Keys to blend (assuming standard MambaTrader output)
        keys = expert_outputs[0].keys()
        
        for key in keys:
            # Stack outputs: (B, n_regimes, ...)
            # Example: direction logits (B, 3)
            stacked = torch.stack([out[key] for out in expert_outputs], dim=1)
            
            # Expand weights to match output shape
            # weights: (B, n_regimes) -> (B, n_regimes, 1, ...)
            w_shape = [-1, self.n_regimes] + [1] * (stacked.dim() - 2)
            w = weights.view(*w_shape)
            
            # Weighted sum
            blended = (stacked * w).sum(dim=1)
            blended_output[key] = blended
            
        return blended_output

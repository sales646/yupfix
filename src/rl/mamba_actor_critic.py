"""
Mamba Actor-Critic for PPO
Wraps MambaTrader to provide policy and value heads.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from typing import Tuple

from ..models.mamba_full import MambaTrader

class MambaActorCritic(nn.Module):
    def __init__(self, mamba_config: dict, action_dim: int = 1, std_init: float = 0.5):
        super().__init__()
        
        # Use MambaTrader as backbone
        # We need to ensure it returns features, not just logits
        self.backbone = MambaTrader(**mamba_config)
        
        # Freeze backbone if needed (optional)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
            
        d_model = mamba_config['d_model']
        
        # Actor Head (Mean of action distribution)
        # Output: [batch, action_dim] (Continuous position size: -1 to 1)
        self.actor_mean = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Bound to [-1, 1]
        )
        
        # Actor Std (Log standard deviation, learnable parameter)
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * np.log(std_init))
        
        # Critic Head (Value function)
        # Output: [batch, 1] (Scalar value)
        self.critic = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        """
        Forward pass returning distribution and value.
        """
        # Get features from backbone
        # Assuming MambaTrader.forward can return features if we modify it or access internal
        # For now, let's assume we can get the last hidden state
        # We might need to adjust MambaTrader to return embedding
        
        # Hack: MambaTrader returns logits. We need the embedding before the head.
        # Let's assume we modify MambaTrader or use its backbone directly.
        # For this implementation, let's assume 'x' is already the embedding or we use the backbone properly.
        
        # If x is raw input (B, L, D), pass through backbone
        features = self.backbone(x, return_features=True) # Need to implement return_features in MambaTrader
        
        # Use last timestep features for RL decision
        # features shape: (B, L, d_model) -> (B, d_model)
        last_features = features[:, -1, :]
        
        # Actor
        action_mean = self.actor_mean(last_features)
        action_std = self.actor_logstd.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        
        # Critic
        value = self.critic(last_features)
        
        return dist, value
    
    def get_action(self, x, deterministic=False):
        dist, value = self(x)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(-1)
        
        return action, log_prob, value
    
    def get_value(self, x):
        _, value = self(x)
        return value

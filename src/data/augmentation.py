"""
Data augmentation for time-series trading data
"""
import torch
import numpy as np
from typing import Tuple

class DataAugmentation:
    """Augmentation techniques to reduce overfitting."""
    
    @staticmethod
    def time_warp(x: torch.Tensor, warp_factor_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape
        device = x.device
        warp_factors = torch.rand(batch_size, device=device) * (warp_factor_range[1] - warp_factor_range[0]) + warp_factor_range[0]
        
        warped = torch.zeros_like(x)
        for b in range(batch_size):
            warp_factor = warp_factors[b].item()
            new_len = max(1, int(seq_len * warp_factor))
            
            if new_len != seq_len:
                temp_indices = torch.linspace(0, seq_len - 1, new_len, device=device).long().clamp(0, seq_len - 1)
                temp_warped = x[b, temp_indices]
                final_indices = torch.linspace(0, temp_warped.shape[0] - 1, seq_len, device=device).long().clamp(0, temp_warped.shape[0] - 1)
                warped[b] = temp_warped[final_indices]
            else:
                warped[b] = x[b]
        return warped
    
    @staticmethod
    def spread_noise(x: torch.Tensor, spread_std: float = 0.0001, spread_prob: float = 0.5) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape
        device = x.device
        mask = torch.rand(batch_size, seq_len, 1, device=device) < spread_prob
        noise = torch.randn(batch_size, seq_len, n_features, device=device) * spread_std
        return x + noise * mask
    
    @staticmethod
    def volatility_scaling(x: torch.Tensor, scale_range: Tuple[float, float] = (0.5, 1.5)) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape
        device = x.device
        scales = torch.rand(batch_size, 1, 1, device=device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        mean = x.mean(dim=1, keepdim=True)
        return mean + (x - mean) * scales
    
    @staticmethod
    def random_shock(x: torch.Tensor, shock_prob: float = 0.05, shock_magnitude: float = 3.0) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape
        device = x.device
        std = x.std(dim=1, keepdim=True) + 1e-8
        shock_mask = torch.rand(batch_size, seq_len, 1, device=device) < shock_prob
        shock_direction = torch.randn(batch_size, seq_len, n_features, device=device).sign()
        shocks = shock_direction * std * shock_magnitude
        return x + shocks * shock_mask
    
    @staticmethod
    def augment_batch(x: torch.Tensor, methods: list = None, p: float = 0.5) -> torch.Tensor:
        if methods is None:
            methods = ['time_warp', 'spread_noise', 'volatility_scaling']
            
        aug_funcs = {
            'time_warp': DataAugmentation.time_warp,
            'spread_noise': DataAugmentation.spread_noise,
            'volatility_scaling': DataAugmentation.volatility_scaling,
            'random_shock': DataAugmentation.random_shock,
        }
        
        # Handle 4D input (B, N, L, F)
        original_shape = x.shape
        if x.dim() == 4:
            B, N, L, F = x.shape
            x_aug = x.view(B * N, L, F)
        else:
            x_aug = x
            
        augmented = x_aug.clone()
        for aug_name in methods:
            if np.random.rand() < p:
                aug_func = aug_funcs.get(aug_name)
                if aug_func:
                    augmented = aug_func(augmented)
        
        # Reshape back if needed
        if len(original_shape) == 4:
            augmented = augmented.view(original_shape)
            
        return augmented

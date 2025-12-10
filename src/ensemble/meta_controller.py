"""
Meta-Controller for Ensemble Signal Blending
Combines ensemble predictions with HMM regime detection for adaptive trading.
"""
import torch
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger("MetaController")


class MetaController:
    """
    Meta-Controller that blends ensemble signals based on HMM regime.
    
    Regime-Based Strategy:
    - STABLE (State 0): High confidence, larger positions, trend-following
    - UNCERTAIN (State 1): Medium confidence, reduced positions, mean-reversion
    - CRISIS (State 2): Low confidence, minimal positions, defensive
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Regime-specific confidence multipliers
        self.regime_confidence = {
            0: 1.0,    # Stable: full confidence
            1: 0.5,    # Uncertain: half confidence
            2: 0.1,    # Crisis: minimal confidence
        }
        
        # Regime-specific position size multipliers
        self.regime_position_scale = {
            0: 1.0,    # Stable: full size
            1: 0.5,    # Uncertain: half size
            2: 0.2,    # Crisis: 20% size (defensive)
        }
        
        # Minimum agreement threshold per regime
        self.regime_agreement_threshold = {
            0: 0.6,    # Stable: 60% agreement required
            1: 0.75,   # Uncertain: 75% agreement (more cautious)
            2: 0.9,    # Crisis: 90% agreement (very cautious)
        }
        
        logger.info("MetaController initialized")
    
    def get_regime(self, hmm_probs: np.ndarray) -> int:
        """
        Determine current market regime from HMM probabilities.
        
        Args:
            hmm_probs: Array of shape (n_states,) with state probabilities
            
        Returns:
            Regime index (0=stable, 1=uncertain, 2=crisis)
        """
        if hmm_probs is None or len(hmm_probs) == 0:
            return 1  # Default to uncertain if no HMM data
            
        return int(np.argmax(hmm_probs))
    
    def blend_signals(self, 
                      ensemble_predictions: Dict,
                      agreement_scores: Dict,
                      hmm_probs: np.ndarray,
                      account_state: Optional[Dict] = None) -> Dict:
        """
        Blend ensemble signals based on regime and agreement.
        
        Args:
            ensemble_predictions: Dict of {horizon: (signal, confidence, uncertainty)}
            agreement_scores: Dict of {horizon: agreement_tensor}
            hmm_probs: HMM state probabilities
            account_state: Optional account info (equity, drawdown, etc.)
            
        Returns:
            Dict with blended signals and meta-info
        """
        regime = self.get_regime(hmm_probs)
        
        conf_multiplier = self.regime_confidence[regime]
        pos_scale = self.regime_position_scale[regime]
        min_agreement = self.regime_agreement_threshold[regime]
        
        blended = {}
        
        for horizon, (signal, confidence, uncertainty) in ensemble_predictions.items():
            if horizon == 'fusion_weights':
                continue
                
            agreement = agreement_scores.get(horizon, torch.ones(signal.shape[0]))
            
            # Scale confidence by regime
            adjusted_confidence = confidence * conf_multiplier
            
            # Scale by inverse uncertainty
            uncertainty_scale = 1.0 / (1.0 + uncertainty * 2.0)
            
            # Final position scale
            final_scale = pos_scale * uncertainty_scale
            
            # Create mask for trades that meet agreement threshold
            trade_mask = (agreement >= min_agreement).float()
            
            blended[horizon] = {
                'signal': signal,
                'confidence': adjusted_confidence,
                'position_scale': final_scale,
                'trade_mask': trade_mask,
                'agreement': agreement,
            }
        
        # Add meta information
        blended['meta'] = {
            'regime': regime,
            'regime_name': ['stable', 'uncertain', 'crisis'][regime],
            'confidence_multiplier': conf_multiplier,
            'position_scale': pos_scale,
            'hmm_probs': hmm_probs.tolist() if isinstance(hmm_probs, np.ndarray) else hmm_probs,
        }
        
        logger.debug(f"Blended signals: regime={regime}, conf_mult={conf_multiplier}")
        
        return blended
    
    def get_final_position(self,
                           blended_signal: Dict,
                           base_position: float,
                           account_state: Optional[Dict] = None) -> float:
        """
        Calculate final position size after all adjustments.
        
        Args:
            blended_signal: Output from blend_signals for one horizon
            base_position: Raw position from model (e.g., -1 to 1)
            account_state: Account info for drawdown-based scaling
            
        Returns:
            Final position size
        """
        position_scale = blended_signal.get('position_scale', 1.0)
        trade_mask = blended_signal.get('trade_mask', 1.0)
        
        # Apply scales
        final_position = base_position * float(position_scale) * float(trade_mask)
        
        # Additional drawdown-based scaling if account state available
        if account_state:
            drawdown = account_state.get('drawdown', 0.0)
            if drawdown > 0.05:  # More than 5% drawdown
                dd_scale = max(0.2, 1.0 - drawdown * 2)  # Reduce size as DD increases
                final_position *= dd_scale
                logger.debug(f"Drawdown scaling: {dd_scale:.2f}")
        
        return final_position
    
    def should_trade(self, blended_signal: Dict, idx: int = 0) -> bool:
        """
        Determine if a trade should be executed based on blended signal.
        
        Args:
            blended_signal: Output from blend_signals for one horizon
            idx: Batch index
            
        Returns:
            True if trade should be executed
        """
        trade_mask = blended_signal.get('trade_mask')
        if trade_mask is None:
            return True
            
        if isinstance(trade_mask, torch.Tensor):
            return bool(trade_mask[idx].item() > 0.5)
        return bool(trade_mask > 0.5)

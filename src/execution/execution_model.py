"""
Realistic Execution Model
Simulates slippage, spread, and fill probabilities.
"""
import numpy as np
import logging

logger = logging.getLogger("ExecutionModel")

class ExecutionModel:
    def __init__(self, avg_slippage_pips=1.0, fill_rate=0.95, random_seed=None):
        """
        Initialize Execution Model.
        
        Args:
            avg_slippage_pips: Average slippage in pips (exponential distribution)
            fill_rate: Probability of order fill (0.0 to 1.0)
            random_seed: Optional seed for reproducibility
        """
        self.slippage = avg_slippage_pips
        self.fill_rate = fill_rate
        if random_seed is not None:
            np.random.seed(random_seed)
            
    def execute(self, signal: int, price: float, spread: float, size: float = 0.0) -> float:
        """
        Simulate order execution.
        
        Args:
            signal: 1 (Buy) or -1 (Sell) (or >0 / <0)
            price: Current mid price
            spread: Current spread
            size: Order size (unused for price calculation but useful for logging/impact)
            
        Returns:
            Executed price or None if rejected
        """
        # Fill probability check
        if np.random.random() > self.fill_rate:
            logger.warning("Order rejected (Fill Rate)")
            return None
            
        # Slippage (exponential distribution)
        # We use exponential because most slippage is small, but tails exist
        actual_slippage = np.random.exponential(self.slippage)
        
        # Directional logic
        # Buy: Ask + Slippage
        # Sell: Bid - Slippage
        
        # Mid price +/- half spread
        half_spread = spread / 2
        
        # Pip size (assuming standard forex 0.0001, JPY 0.01 needs handling if generic)
        # Heuristic for pip size:
        if price > 50: # JPY pairs, Indices
            pip_size = 0.01
        else: # Standard Forex
            pip_size = 0.0001
            
        slippage_value = actual_slippage * pip_size
        
        if signal > 0:  # BUY
            # Executed at Ask + Slippage
            exec_price = price + half_spread + slippage_value
        else:  # SELL
            # Executed at Bid - Slippage
            exec_price = price - half_spread - slippage_value
            
        return exec_price

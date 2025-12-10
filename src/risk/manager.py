"""
Guardian Risk Manager with Profit Scaling
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RiskManager:
    """Guardian Risk Manager - immutable safety rules."""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_drawdown = config.get('max_drawdown', 0.08)
        self.max_position_size = config.get('max_position_size', 1.0)
        self.max_daily_loss = config.get('max_daily_loss', 0.04)
        
        self.profit_scaling = config.get('position_sizing', {}).get('profit_scaling', {})
        self.scaling_enabled = self.profit_scaling.get('enabled', False)
        self.scaling_levels = self.profit_scaling.get('levels', [])
        self.scaling_levels.sort(key=lambda x: x['profit_percent'], reverse=True)
        
        # Daily tracking
        self.daily_start_balance = None
        self.last_reset_date = None
        
        # Correlation groups
        self.correlation_groups = config.get('correlation_groups', {})
        
        logger.info(f"RiskManager: Max DD={self.max_drawdown}, Scaling={self.scaling_enabled}")
        
    def check_position(self, target_position: float, balance: float, 
                       peak_balance: float, initial_balance: float = None,
                       current_date: str = None) -> float:
        # Reset daily tracking at new day
        if current_date and current_date != self.last_reset_date:
            self.daily_start_balance = balance
            self.last_reset_date = current_date
        
        # 1. Calculate Drawdown
        current_dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        
        # 2. Check Daily DD
        if self.daily_start_balance:
            daily_dd = (self.daily_start_balance - balance) / self.daily_start_balance
            if daily_dd >= self.max_daily_loss:
                logger.warning(f"DAILY STOP: DD {daily_dd:.2%} >= Max {self.max_daily_loss:.2%}")
                return 0.0
        
        # 3. Hard Stop (Total DD)
        if current_dd >= self.max_drawdown:
            logger.warning(f"HARD STOP: DD {current_dd:.2%} >= Max {self.max_drawdown:.2%}")
            return 0.0
        
        # 4. Cap Position
        allowed_position = min(abs(target_position), self.max_position_size)
        if target_position < 0:
            allowed_position = -allowed_position
            
        # 5. Scale Down Near Max DD
        dd_threshold = self.max_drawdown * 0.5
        if current_dd > dd_threshold:
            safety_margin = (self.max_drawdown - current_dd) / (self.max_drawdown - dd_threshold)
            allowed_position *= safety_margin
            
        # 6. Profit Scaling
        if self.scaling_enabled and initial_balance is None:
            logger.warning("Profit scaling enabled but initial_balance not provided")
            
        if self.scaling_enabled and initial_balance and balance > initial_balance:
            profit_pct = (balance - initial_balance) / initial_balance * 100
            multiplier = 1.0
            
            for level in self.scaling_levels:
                if profit_pct >= level['profit_percent']:
                    multiplier = level['multiplier']
                    break
            
            if multiplier != 1.0:
                allowed_position *= multiplier
            
        return allowed_position
    
    def check_correlation_exposure(self, proposed_positions: Dict[str, float]) -> Dict[str, float]:
        """Limit exposure per correlation group"""
        max_group_exposure = 0.04  # 4% as default
        
        adjusted = proposed_positions.copy()
        
        for group_name, symbols in self.correlation_groups.items():
            group_exposure = sum(
                abs(adjusted.get(sym.replace('_LONG', '').replace('_SHORT', ''), 0))
                for sym in symbols
            )
            
            if group_exposure > max_group_exposure:
                scale = max_group_exposure / group_exposure
                logger.info(f"Scaling {group_name} group from {group_exposure:.2%} to {max_group_exposure:.2%}")
                for sym in symbols:
                    base_sym = sym.replace('_LONG', '').replace('_SHORT', '')
                    if base_sym in adjusted:
                        adjusted[base_sym] *= scale
        
        return adjusted

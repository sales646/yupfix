"""
Guardian Risk Manager
Enforces FTMO trading limits and risk controls.
"""
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger("RiskManager")


class RiskManager:
    """
    Risk Manager enforcing FTMO Challenge limits.
    
    Limits:
    - Daily Loss Limit: 5% of initial balance
    - Max Loss Limit: 10% of initial balance
    - Profit Target: 10% for Phase 1, 5% for Phase 2
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # FTMO Limits (with defaults)
        risk_config = config.get('risk', {})
        self.daily_loss_limit_pct = risk_config.get('daily_loss_limit_pct', 0.05)
        self.max_loss_limit_pct = risk_config.get('max_loss_limit_pct', 0.10)
        self.profit_target_pct = risk_config.get('profit_target_pct', 0.10)
        self.buffer = risk_config.get('max_drawdown_buffer', 0.005)
        
        # Account State
        self.initial_balance = 100000.0
        self.current_equity = 100000.0
        self.start_of_day_equity = 100000.0
        self.peak_equity = 100000.0
        
        # Tracking
        self.trading_days = set()
        self.start_date = datetime.now()
        self.scaling_profit_accumulated = 0.0
        self.payouts_count = 0
        
        # Flags
        self.kill_switch_active = False
        self.daily_stop_active = False
        self.payout_lock_active = False
        self.is_trading_disabled = False
        self.fail_reason = ""
        
        logger.info(f"RiskManager initialized: Daily={self.daily_loss_limit_pct:.1%}, Max={self.max_loss_limit_pct:.1%}")

    def update_equity(self, new_equity: float):
        """Update current equity and track peak."""
        self.current_equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)
        
    def reset_daily(self):
        """Reset daily tracking (call at start of each trading day)."""
        self.start_of_day_equity = self.current_equity
        self.daily_stop_active = False
        logger.info(f"Daily reset. Start of day equity: {self.start_of_day_equity:.2f}")
        
    def get_daily_drawdown(self) -> float:
        """Calculate current daily drawdown percentage."""
        if self.start_of_day_equity <= 0:
            return 0.0
        return (self.start_of_day_equity - self.current_equity) / self.start_of_day_equity
    
    def get_total_drawdown(self) -> float:
        """Calculate total drawdown from initial balance."""
        if self.initial_balance <= 0:
            return 0.0
        return (self.initial_balance - self.current_equity) / self.initial_balance
    
    def get_profit_pct(self) -> float:
        """Calculate current profit percentage."""
        if self.initial_balance <= 0:
            return 0.0
        return (self.current_equity - self.initial_balance) / self.initial_balance

    def check_trade_allowed(self, signal: Dict) -> bool:
        """
        Validate if a new trade is allowed.
        
        Args:
            signal: Trade signal dict with 'symbol', 'direction', 'size'
            
        Returns:
            True if trade is allowed, False otherwise
        """
        # Kill switch check
        if self.kill_switch_active or self.is_trading_disabled:
            logger.warning(f"Trade Rejected: Kill Switch Active - {self.fail_reason}")
            return False
            
        if self.daily_stop_active:
            logger.warning("Trade Rejected: Daily Stop Active")
            return False
            
        if self.payout_lock_active:
            logger.warning("Trade Rejected: Payout Lock-in Active")
            return False
            
        # Daily Loss Limit Check
        daily_dd = self.get_daily_drawdown()
        if daily_dd > (self.daily_loss_limit_pct - self.buffer):
            logger.warning(f"Trade blocked: Daily DD {daily_dd:.2%} >= limit {self.daily_loss_limit_pct:.2%}")
            self.daily_stop_active = True
            return False

        # Max Loss Limit Check
        total_dd = self.get_total_drawdown()
        if total_dd > (self.max_loss_limit_pct - self.buffer):
            logger.warning(f"Trade blocked: Total DD {total_dd:.2%} >= limit {self.max_loss_limit_pct:.2%}")
            self.trigger_kill_switch("Max Loss Limit Breached")
            return False
            
        return True
    
    def check_position(self, target_weight: float, balance: float, 
                       peak_equity: float, initial_balance: float) -> float:
        """
        Check and possibly reduce position size based on risk limits.
        
        Args:
            target_weight: Desired position weight [-1, 1]
            balance: Current balance
            peak_equity: Peak equity reached
            initial_balance: Starting balance
            
        Returns:
            Adjusted position weight (may be reduced or zeroed)
        """
        # If near daily limit, reduce size
        daily_dd = self.get_daily_drawdown()
        if daily_dd > self.daily_loss_limit_pct * 0.8:
            # Scale down position as we approach limit
            scale_factor = max(0, 1.0 - (daily_dd / self.daily_loss_limit_pct))
            return target_weight * scale_factor
            
        return target_weight
    
    def check_correlation_exposure(self, positions: Dict[str, float]) -> Dict[str, float]:
        """
        Check and reduce positions if correlation exposure is too high.
        
        Args:
            positions: Dict of {symbol: weight}
            
        Returns:
            Adjusted positions dict
        """
        # Simple implementation: cap total exposure
        max_total_exposure = 3.0  # Max 300% total exposure
        total_exposure = sum(abs(w) for w in positions.values())
        
        if total_exposure > max_total_exposure:
            scale = max_total_exposure / total_exposure
            return {sym: w * scale for sym, w in positions.items()}
            
        return positions

    def record_trade(self, trade: Dict):
        """Log trade to track trading days."""
        today = datetime.now().date()
        self.trading_days.add(today)
        logger.info(f"Trade Recorded. Active Trading Days: {len(self.trading_days)}")

    def check_scaling_eligibility(self) -> bool:
        """
        Check if account is eligible for FTMO scaling (10% profit over 4 months).
        """
        months_active = (datetime.now() - self.start_date).days / 30
        profit_pct = self.get_profit_pct()
        
        if months_active >= 4 and profit_pct >= 0.10 and self.payouts_count >= 2:
            return True
        return False

    def trigger_kill_switch(self, reason: str):
        """Activate kill switch and disable all trading."""
        self.is_trading_disabled = True
        self.kill_switch_active = True
        self.fail_reason = reason
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
        # In production: send CLOSE_ALL command to bridge
        
    def get_status(self) -> Dict:
        """Get current risk status summary."""
        return {
            "equity": self.current_equity,
            "daily_drawdown": self.get_daily_drawdown(),
            "total_drawdown": self.get_total_drawdown(),
            "profit_pct": self.get_profit_pct(),
            "trading_days": len(self.trading_days),
            "kill_switch": self.kill_switch_active,
            "daily_stop": self.daily_stop_active,
        }

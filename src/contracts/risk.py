"""
Risk Contracts
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from datetime import date

class Position(BaseModel):
    """Current position state"""
    symbol: str
    size: float  # positive=long, negative=short
    entry_price: float = Field(gt=0)
    current_price: float = Field(gt=0)
    unrealized_pnl: float
    
    @property
    def direction(self) -> int:
        return 1 if self.size > 0 else -1 if self.size < 0 else 0

class PortfolioState(BaseModel):
    """Full portfolio state - Guardian input"""
    balance: float = Field(gt=0)
    peak_balance: float = Field(gt=0)
    initial_balance: float = Field(gt=0)
    current_date: date
    positions: Dict[str, Position] = {}
    
    @property
    def current_dd(self) -> float:
        return (self.peak_balance - self.balance) / self.peak_balance
    
    @property
    def daily_dd(self) -> float:
        # Requires daily_start_balance tracking
        return 0.0

class RiskDecision(BaseModel):
    """Guardian output"""
    allowed: bool
    original_size: float
    adjusted_size: float
    reason: Optional[str] = None
    
    # Limits hit
    daily_dd_hit: bool = False
    total_dd_hit: bool = False
    correlation_adjusted: bool = False
    profit_scaled: bool = False

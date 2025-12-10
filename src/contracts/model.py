"""
Model Contracts
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional, Tuple, Any

class Prediction(BaseModel):
    """Single prediction output"""
    symbol: str
    direction: int = Field(ge=-1, le=1)  # -1=short, 0=flat, 1=long
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0)
    magnitude: float = Field(ge=0, le=1)
    
    class Config:
        arbitrary_types_allowed = True

class TradeSignal(BaseModel):
    """Actionable trade signal - PPO output"""
    symbol: str
    action: int = Field(ge=-1, le=1)  # -1=sell, 0=hold, 1=buy
    position_size: float = Field(ge=0, le=1)  # fraction of capital
    confidence: float = Field(ge=0, le=1)
    regime: int = Field(ge=0, le=2)
    
    # Risk-adjusted
    stop_loss_pips: Optional[float] = None
    take_profit_pips: Optional[float] = None
    
    class Config:
        frozen = True

"""
Data Contracts
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

class OHLCVBar(BaseModel):
    """Single OHLCV bar - boundary contract"""
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    
    # Optional microstructure
    buy_volume: Optional[float] = Field(ge=0, default=None)
    sell_volume: Optional[float] = Field(ge=0, default=None)
    bid_close: Optional[float] = Field(gt=0, default=None)
    ask_close: Optional[float] = Field(gt=0, default=None)
    spread_avg: Optional[float] = Field(ge=0, default=None)
    spread_max: Optional[float] = Field(ge=0, default=None)
    tick_count: Optional[int] = Field(ge=0, default=None)
    
    @validator('high')
    def high_gte_low(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('high must be >= low')
        return v

    class Config:
        frozen = True  # Immutable


class FeatureRow(BaseModel):
    """24 technical features - output contract"""
    timestamp: datetime
    
    # Price/Volume
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Returns/Volatility
    log_return: float
    hl_range: float
    atr_14: float
    
    # Trend
    sma_20: float
    sma_50: float
    ema_20: float
    ema_50: float
    macd_line: float
    macd_signal: float
    
    # Momentum
    rsi_14: float
    momentum_10: float
    roc_10: float
    cci_14: float
    
    # Bands
    bb_upper: float
    bb_middle: float
    bb_lower: float
    
    # Other
    vwap: float
    tick_count: float
    spread: float
    
    class Config:
        frozen = True


class LabelRow(BaseModel):
    """Training labels - output contract"""
    timestamp: datetime
    direction: int = Field(ge=0, le=2)      # 0=down, 1=flat, 2=up
    volatility: int = Field(ge=0, le=2)     # 0=low, 1=med, 2=high
    magnitude: float = Field(ge=0, le=1)    # normalized
    
    class Config:
        frozen = True

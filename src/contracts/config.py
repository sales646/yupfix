"""
Configuration Contracts
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum

class AugmentationMethod(str, Enum):
    TIME_WARP = "time_warp"
    SPREAD_NOISE = "spread_noise"
    VOLATILITY_SCALING = "volatility_scaling"
    RANDOM_SHOCK = "random_shock"

class ModelConfig(BaseModel):
    """Mamba model configuration"""
    d_input: int = Field(default=24, ge=1)
    d_model: int = Field(default=256, ge=32)
    n_layers: int = Field(default=12, ge=1)
    d_state: int = Field(default=128, ge=16)
    n_assets: int = Field(default=6, ge=1)
    expand: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.1, ge=0, le=0.5)
    use_multi_label: bool = True

class TrainingConfig(BaseModel):
    """Training configuration - YAML contract"""
    batch_size: int = Field(default=16, ge=1)
    epochs: int = Field(default=50, ge=1)
    learning_rate: float = Field(default=0.0001, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0)
    
    @validator('learning_rate')
    def lr_reasonable(cls, v):
        if v > 0.1:
            raise ValueError('learning_rate too high, likely a mistake')
        return v

class RiskConfig(BaseModel):
    """Risk limits - Guardian contract"""
    daily_drawdown_percent: float = Field(default=4.0, gt=0, le=10)
    total_drawdown_percent: float = Field(default=8.0, gt=0, le=15)
    max_risk_per_trade_percent: float = Field(default=1.25, gt=0, le=5)
    max_concurrent_trades: int = Field(default=4, ge=1, le=20)
    
    @validator('total_drawdown_percent')
    def total_gte_daily(cls, v, values):
        if 'daily_drawdown_percent' in values and v < values['daily_drawdown_percent']:
            raise ValueError('total_drawdown must be >= daily_drawdown')
        return v

"""
YUP-250 Data Contracts
Strict Pydantic models for system boundaries.
"""
from .config import TrainingConfig, ModelConfig, RiskConfig
from .data import OHLCVBar, FeatureRow, LabelRow
from .model import Prediction, TradeSignal
from .risk import Position, PortfolioState, RiskDecision
from .pipeline import TrainingBatch, PipelineState

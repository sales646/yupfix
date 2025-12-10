"""
Pipeline Contracts
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class TrainingBatch(BaseModel):
    """Batch of data for training"""
    X: Any # torch.Tensor or np.ndarray
    y: Any
    symbols: List[str]
    
    class Config:
        arbitrary_types_allowed = True

class PipelineState(BaseModel):
    """Current state of the pipeline"""
    current_epoch: int = 0
    current_step: int = 0
    best_val_loss: float = float('inf')
    is_training: bool = False

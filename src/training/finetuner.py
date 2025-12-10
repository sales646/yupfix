"""
Weekly Finetuner
Handles incremental training on new data with transfer learning.
"""
import torch
import torch.optim as optim
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional

from .trainer import train_epoch, save_checkpoint

logger = logging.getLogger("WeeklyFinetuner")

class WeeklyFinetuner:
    """
    Finetunes the model on the most recent data (e.g., last week).
    Uses transfer learning by freezing early layers.
    """
    def __init__(self, model, config: dict, save_dir: str = "models/finetuned"):
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.last_update: Optional[datetime] = None
        
    def should_update(self, current_date: datetime) -> bool:
        """Check if 7 days have passed since last update"""
        if self.last_update is None:
            return True
        days_since = (current_date - self.last_update).days
        return days_since >= 7
        
    def finetune(self, dataloader, current_date: datetime):
        """
        Perform incremental training.
        
        Args:
            dataloader: DataLoader containing new data (e.g., last week)
            current_date: Current simulation/real-world date
        """
        logger.info(f"Starting Weekly Finetuning for {current_date.date()}")
        
        # 1. Freeze early layers (Transfer Learning)
        # Assuming model has a 'backbone' with 'layers'
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layers'):
            n_layers = len(self.model.backbone.layers)
            n_freeze = min(8, n_layers) # Freeze first 8 layers (or all if < 8)
            
            logger.info(f"Freezing first {n_freeze} layers of backbone")
            for i, layer in enumerate(self.model.backbone.layers):
                if i < n_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
        else:
            logger.warning("Model structure unknown, skipping layer freezing")
            
        # 2. Setup Optimizer with lower LR
        # Only optimize parameters that require grad
        finetune_lr = self.config['training'].get('finetune_lr', 1e-5) # Default 10x lower than 1e-4
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=finetune_lr,
            weight_decay=self.config['training']['weight_decay']
        )
        
        # 3. Short Training Loop
        epochs = self.config['training'].get('finetune_epochs', 5)
        
        for epoch in range(epochs):
            loss = train_epoch(
                self.model, 
                dataloader, 
                optimizer, 
                self.config, 
                epoch=epoch, 
                writer=None # No tensorboard for finetuning to keep it simple
            )
            logger.info(f"Finetune Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
            
        # 4. Unfreeze all layers for future use (or keep frozen? Usually unfreeze)
        # If we keep the model instance alive, we should probably unfreeze 
        # so that the next finetune starts from a clean state of "everything trainable" 
        # before we freeze again. Or we just leave them frozen if we want permanent freezing.
        # Let's unfreeze to be safe.
        for param in self.model.parameters():
            param.requires_grad = True
            
        # 5. Save Finetuned Model
        date_str = current_date.strftime("%Y%m%d")
        save_path = self.save_dir / f"finetuned_{date_str}.pt"
        save_checkpoint(self.model, optimizer, None, epochs, loss, save_path)
        
        self.last_update = current_date
        logger.info(f"Finetuning complete. Model saved to {save_path}")

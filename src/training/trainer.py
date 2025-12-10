"""
Shared Training Logic
Contains core training loops and loss functions used by both initial training and fine-tuning.
"""
import torch
import torch.nn.functional as F
import logging
from src.data.augmentation import DataAugmentation
from src.training.risk_aware_loss import DrawdownAwareLoss

logger = logging.getLogger(__name__)

def compute_loss(outputs, y, config):
    """
    Proper multi-task loss computation.
    Handles both single-asset (B, L, C) and multi-asset (B, N, L, C) inputs.
    """
    total_loss = 0
    loss_weights = config['loss']
    ml_config = config.get('multi_label', {})
    
    # Handle Multi-Asset Input (B, N, L, C) -> Flatten to (B*N, L, C)
    if y.dim() == 4:
        B, N, L, C = y.shape
        y = y.view(B * N, L, C)
        
        # Flatten outputs similarly
        if isinstance(outputs, dict):
            new_outputs = {}
            for k, v in outputs.items():
                if k == 'fusion_weights':
                    new_outputs[k] = v 
                    continue
                    
                if isinstance(v, torch.Tensor):
                    # (B, N, ...) -> (B*N, ...)
                    new_outputs[k] = v.view(-1, *v.shape[2:])
                elif isinstance(v, tuple):
                    # Tuple of tensors
                    new_outputs[k] = tuple(t.view(-1, *t.shape[2:]) for t in v)
                else:
                    new_outputs[k] = v
            outputs = new_outputs
        elif isinstance(outputs, torch.Tensor):
             outputs = outputs.view(-1, *outputs.shape[2:])
    
    # Extract targets from y (assuming y shape: B, L, features)
    # Features: [direction, volatility, magnitude, ...]
    # Take last timestep for prediction
    if y.dim() == 3 and y.shape[2] >= 3:
        direction_target = y[:, -1, 0].long()      # Direction: 0=down, 1=neutral, 2=up
        volatility_target = y[:, -1, 1].long()     # Volatility: low/med/high
        magnitude_target = y[:, -1, 2].float()     # Magnitude: continuous
    else:
        # Fallback: use first column as direction
        direction_target = y[:, -1, 0].long() if y.dim() == 3 else y[:, -1].long()
        volatility_target = direction_target.clone()
        magnitude_target = torch.zeros_like(direction_target).float()
    
    if isinstance(outputs, dict):
        # Multi-horizon outputs
        for horizon, value in outputs.items():
            if horizon == 'fusion_weights':
                continue
            
            if horizon == 'multi_label':
                # Multi-label head
                direction, volatility, magnitude, confidence, uncertainty = value
                
                ml_loss = (
                    ml_config.get('direction_weight', 1.0) * F.cross_entropy(direction, direction_target) +
                    ml_config.get('volatility_weight', 0.5) * F.cross_entropy(volatility, volatility_target) +
                    ml_config.get('magnitude_weight', 0.3) * F.mse_loss(magnitude.squeeze(), magnitude_target)
                )
                total_loss += ml_loss
            else:
                # Horizon output (scalp, intraday, swing)
                signal, conf, unc = value
                
                # Direction loss
                direction_loss = F.cross_entropy(signal, direction_target)
                
                # Confidence loss (BCE - high conf when correct)
                correct = (signal.argmax(dim=-1) == direction_target).float()
                conf_loss = F.binary_cross_entropy(conf.squeeze(), correct)
                
                # Get horizon weight
                # Map horizon names to config keys if needed
                # config keys: horizon_1min, horizon_5min, etc.
                # output keys: scalp, intraday, swing
                # Mapping: scalp->1min, intraday->1hour, swing->4hour?
                # Let's check config.
                # Config has: horizon_1min, horizon_5min, horizon_15min, horizon_1hour.
                # But heads are: scalp, intraday, swing.
                # We should probably update config or mapping.
                # Let's map: scalp->horizon_1min, intraday->horizon_1hour, swing->horizon_15min (approx)
                # Or just use default 1.0 if not found.
                
                weight_key = f'horizon_{horizon}' # e.g. horizon_scalp
                # Try to map common names
                if horizon == 'scalp': weight_key = 'horizon_1min'
                elif horizon == 'intraday': weight_key = 'horizon_1hour'
                elif horizon == 'swing': weight_key = 'horizon_4hour' # Not in config, use default
                
                weight = loss_weights.get(weight_key, 1.0)
                
                total_loss += weight * (direction_loss + 0.2 * conf_loss)
    else:
        # Single output (fallback)
        if isinstance(outputs, tuple):
            signal = outputs[0]
        else:
            signal = outputs
        total_loss = F.cross_entropy(signal, direction_target)
    
    return total_loss


def train_epoch(model, dataloader, optimizer, config, epoch, writer=None, risk_aware=True):
    """Train one epoch with optional risk-aware loss.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        config: Training config
        epoch: Current epoch number
        writer: TensorBoard writer
        risk_aware: If True, use drawdown-aware loss
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    gradient_clip = config['training']['gradient_clip']
    accumulation_steps = config['training'].get('accumulation_steps', 1)
    log_every = config['logging']['log_every']
    
    # Initialize Augmenter
    augmenter = DataAugmentation()
    aug_config = config.get('augmentation', {})
    use_aug = aug_config.get('enabled', False)
    
    # Initialize Risk-Aware Loss
    risk_loss_fn = None
    if risk_aware:
        risk_config = config.get('risk_loss', {
            'daily_loss_limit': 0.05,
            'max_loss_limit': 0.10,
            'drawdown_penalty': 2.0,
            'volatility_penalty': 0.5,
            'consistency_bonus': 0.3
        })
        risk_loss_fn = DrawdownAwareLoss(risk_config)
        risk_loss_fn.reset_tracking()
    
    # Track risk metrics
    total_dd_penalty = 0
    total_vol_penalty = 0
    
    for batch_idx, (X, y, symbols) in enumerate(dataloader):
        if batch_idx == 0:
            print("DEBUG: Entered training loop")
            
        # Move to device
        device = next(model.parameters()).device
        X = X.to(device)
        y = y.to(device)
        
        if batch_idx == 0:
            print(f"DEBUG: Data moved to {device}, X shape: {X.shape}")
        
        # Apply Augmentation
        if use_aug and model.training:
            X = augmenter.augment_batch(
                X,
                methods=aug_config.get('methods', ['noise', 'warp']),
                p=aug_config.get('probability', 0.5)
            )
        
        # Forward pass
        if batch_idx == 0:
            print("DEBUG: Starting forward pass...")
        outputs = model(X)
        if batch_idx == 0:
            print("DEBUG: Forward pass complete")
        
        # Compute loss using proper loss function
        base_loss = compute_loss(outputs, y, config)
        
        # Add risk-aware penalty if enabled
        if risk_loss_fn is not None:
            loss, loss_breakdown = risk_loss_fn(outputs, y, base_loss)
            total_dd_penalty += loss_breakdown.get('drawdown_penalty', 0)
            total_vol_penalty += loss_breakdown.get('volatility_penalty', 0)
        else:
            loss = base_loss
        
        # Normalize by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # Logging
        if (batch_idx + 1) % log_every == 0:
            avg_loss = total_loss / num_batches
            avg_dd = total_dd_penalty / num_batches if risk_loss_fn else 0
            avg_vol = total_vol_penalty / num_batches if risk_loss_fn else 0
            
            log_msg = f"Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}"
            if risk_loss_fn:
                log_msg += f", DD_Penalty: {avg_dd:.4f}, Vol_Penalty: {avg_vol:.4f}"
            logger.info(log_msg)
            
            if writer:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Loss/train_batch', loss.item() * accumulation_steps, global_step)
                if risk_loss_fn:
                    writer.add_scalar('Risk/drawdown_penalty', avg_dd, global_step)
                    writer.add_scalar('Risk/volatility_penalty', avg_vol, global_step)
    
    avg_loss = total_loss / num_batches
    
    # Return metrics dict
    metrics = {
        'loss': avg_loss,
        'drawdown_penalty': total_dd_penalty / num_batches if risk_loss_fn and num_batches > 0 else 0,
        'volatility_penalty': total_vol_penalty / num_batches if risk_loss_fn and num_batches > 0 else 0
    }
    return avg_loss  # Keep backward compatible, metrics logged via TensorBoard


def validate(model, dataloader, config, epoch, writer=None):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for X, y, symbols in dataloader:
            device = next(model.parameters()).device
            X = X.to(device)
            y = y.to(device)
            
            # Forward pass
            outputs = model(X)
            
            # Compute loss using proper loss function
            loss = compute_loss(outputs, y, config)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    if writer:
        writer.add_scalar('Loss/val', avg_loss, epoch)
    
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path}")

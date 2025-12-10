"""
Yup250 Training Script
Main entry point for training the Mamba trading system
"""
import argparse
import yaml
import torch
import torch.optim as optim
from pathlib import Path
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yup250_pipeline import Yup250Pipeline
from src.targets.label_generator import create_labels_for_symbols, align_features_and_labels
from src.data.augmentation import DataAugmentation
from src.training.trainer import train_epoch, validate, save_checkpoint, compute_loss
from src.contracts.config import TrainingConfig
from src.contracts.pipeline_validator import validate_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Validate training part of config
    # Note: The full config structure is complex, we validate the 'training' section here
    # or we could define a FullConfig contract. For now, let's validate what we have contracts for.
    if 'training' in raw_config:
        TrainingConfig(**raw_config['training'])
        
    return raw_config


def create_optimizer(model, config: dict):
    """Create optimizer from config"""
    opt_config = config['optimizer']
    
    if opt_config['name'] == 'AdamW':
        return optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=opt_config['betas'],
            eps=opt_config['eps'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['name']}")


def create_scheduler(optimizer, config: dict):
    """Create learning rate scheduler"""
    sched_config = config['scheduler']
    
    if sched_config['name'] == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config['T_max'],
            eta_min=sched_config['eta_min']
        )
    elif sched_config['name'] == 'CosineAnnealingWarmRestarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_config['T_0'],
            T_mult=sched_config.get('T_mult', 1),
            eta_min=sched_config.get('eta_min', 0)
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_config['name']}")





def main(args):
    # Load config
    config = load_config(args.config)
    
    # Override config with args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Setup logging directory
    log_dir = Path(config['logging']['log_dir'])
    save_dir = Path(config['logging']['save_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = None
    if config['logging'].get('tensorboard', True):
        tb_dir = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(tb_dir)
        logger.info(f"TensorBoard logging to: {tb_dir}")
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = Yup250Pipeline(config)
    
    # Load training data
    logger.info("Loading training data...")
    train_data = pipeline.load_data(config['data']['train_path'])
    
    # Fit HMM
    logger.info("Fitting HMM...")
    pipeline.fit_hmm(train_data)
    
    # Prepare features
    logger.info("Preparing features...")
    train_features = pipeline.prepare_features(train_data)
    
    # Save scalers immediately
    pipeline.save_scalers(save_dir)
    
    # Create model
    logger.info("Creating model...")
    use_ensemble = config.get('ensemble', {}).get('enabled', False)
    n_models = config.get('ensemble', {}).get('n_models', 3)
    model = pipeline.create_model(use_ensemble=use_ensemble, n_models=n_models)
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Using device: {device}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Create proper labels using label generator
    logger.info("Creating labels...")
    train_labels = create_labels_for_symbols(
        train_data, 
        horizon=args.horizon if hasattr(args, 'horizon') else 12,
        delay=args.delay if hasattr(args, 'delay') else 3
    )
    
    # Align features and labels
    logger.info("Aligning features and labels...")
    train_features, train_labels = align_features_and_labels(train_features, train_labels)
    
    # ✅ VALIDATE PIPELINE BEFORE TRAINING
    logger.info("\n" + "="*50)
    logger.info("VALIDATING PIPELINE...")
    logger.info("="*50)
    
    validate_pipeline(
        config=config,
        features=train_features,
        labels=train_labels,
        model=model
    )
    
    logger.info("="*50 + "\n")
    
    # Create dataloader
    batch_size = config['training']['batch_size']
    train_loader = pipeline.create_dataloader(train_features, train_labels, batch_size, shuffle=True)
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    
    # Load validation data if specified
    val_loader = None
    if args.val_path:
        logger.info("Loading validation data...")
        val_data = pipeline.load_data(args.val_path)
        val_features = pipeline.prepare_features(val_data)
        
        # Create and align validation labels
        logger.info("Creating validation labels...")
        val_labels = create_labels_for_symbols(
            val_data,
            horizon=args.horizon if hasattr(args, 'horizon') else 12,
            delay=args.delay if hasattr(args, 'delay') else 3
        )
        val_features, val_labels = align_features_and_labels(val_features, val_labels)
        
        val_loader = pipeline.create_dataloader(val_features, val_labels, batch_size, shuffle=False)
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Training loop
    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, start_epoch + epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, config, epoch, writer)
        logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
        
        if writer:
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Scheduler step
        if scheduler:
            scheduler.step()
        
        # Validation
        if val_loader and epoch % config['validation']['val_every'] == 0:
            val_loss = validate(model, val_loader, config, epoch, writer)
            logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if config['validation']['early_stopping']['enabled']:
                min_delta = config['validation']['early_stopping']['min_delta']
                patience = config['validation']['early_stopping']['patience']
                
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_path = save_dir / "best_model.pt"
                    save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_path)
                    logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
                    
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
        
        # Save checkpoint
        if epoch % config['logging']['save_every'] == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, checkpoint_path)
    
    # Save final model
    final_path = save_dir / "final_model.pt"
    save_checkpoint(model, optimizer, scheduler, epochs, train_loss, final_path)
    
    logger.info("\nTraining completed!")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Yup250 Mamba Trading System')
    parser.add_argument('--config', type=str, default='config/training.yaml',
                       help='Path to training config YAML')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--val-path', type=str, default=None,
                       help='Path to validation data')
    parser.add_argument('--horizon', type=int, default=12,
                       help='Target horizon in bars (default: 12)')
    parser.add_argument('--delay', type=int, default=3,
                       help='Delay zone bars (default: 3)')
    
    args = parser.parse_args()
    main(args)

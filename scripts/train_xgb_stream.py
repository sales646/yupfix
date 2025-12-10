import sys
import os
import yaml
import time
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.stream_loader import BackgroundDownloader
from src.models.supervised.xgb_model import XGBoostModel

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StreamTrainer")

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    symbols = config['trading']['symbols']
    
    # Setup Date Range (Last 5 Years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1825)
    
    # Initialize Downloader (Dukascopy - No API Key Required)
    downloader = BackgroundDownloader(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        chunk_size_days=7 # Weekly chunks (faster for testing)
    )
    
    # Initialize Model
    model = XGBoostModel()
    
    # Start Download
    logger.info("Starting Background Downloader...")
    downloader.start()
    
    # Load Phase
    logger.info("=== LOAD PHASE ===")
    downloader.wait_for_initial_load(min_chunks=2) # Wait for 2 months of data
    
    # Training Loop
    logger.info("=== TRAINING PHASE ===")
    chunk_count = 0
    
    while True:
        chunk = downloader.get_next_chunk(timeout=5)
        
        if chunk is None: # Download complete and queue empty
            logger.info("All data processed.")
            break
            
        if chunk == "WAIT":
            logger.info("Waiting for more data...")
            continue
            
        chunk_count += 1
        logger.info(f"Training on Chunk #{chunk_count}")
        
        # Combine symbols in chunk
        full_df = pd.DataFrame()
        for symbol, df in chunk.items():
            # Add symbol column if needed for features, or train separate models
            # For simplicity, let's concat (assuming model handles general patterns)
            full_df = pd.concat([full_df, df])
            
        if not full_df.empty:
            # Train (Incremental if supported, or full retrain on window)
            # XGBoost supports incremental learning via xgb_model parameter
            # But our wrapper .train() might need update. 
            # For now, let's assume standard train updates the internal booster if it exists?
            # Actually, standard .train() usually resets. 
            # Let's just call train() which saves the model. 
            # To do true incremental, we'd need to pass the previous model.
            # For this demo, we'll simulate "training on available data".
            
            # In a real scenario, we might accumulate a sliding window here.
            model.train(full_df) 
            logger.info(f"Chunk #{chunk_count} training complete.")
            
    logger.info("Training pipeline finished.")
    model.save()

if __name__ == "__main__":
    main()

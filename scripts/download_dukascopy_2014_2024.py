"""
Download Historical Data from Dukascopy (2014-2024)
Fetches 10 years of tick data for forex pairs
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dukascopy_loader import DukascopyLoader
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DukascopyDownload")

def download_historical_data():
    """
    Download 10 years of data (2014-2024) for all forex symbols.
    
    Note: Dukascopy only has forex pairs, not indices like NAS100, US30.
    """
    loader = DukascopyLoader()
    
    # Symbols (only forex available on Dukascopy)
    forex_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
    
    # Date range: 2014-2024 (10 years)
    start_date = "2014-01-01"
    end_date = "2024-12-31"
    
    logger.info(f"Starting download: {start_date} to {end_date}")
    logger.info(f"Symbols: {', '.join(forex_symbols)}")
    logger.info("="*60)
    
    for symbol in forex_symbols:
        logger.info(f"\n📥 Downloading {symbol}...")
        logger.info(f"   Period: {start_date} to {end_date}")
        
        try:
            # Fetch data with 1-minute resampling
            df = loader.fetch_data(symbol, start_date, end_date, resample='1min')
            
            if df is not None and not df.empty:
                # Save to raw directory
                os.makedirs("data/raw", exist_ok=True)
                filename = f"data/raw/{symbol}_1m_{start_date}_{end_date}.parquet"
                df.to_parquet(filename)
                
                logger.info(f"✅ {symbol}: {len(df)} candles saved to {filename}")
                logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
                logger.info(f"   Size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
            else:
                logger.warning(f"❌ {symbol}: No data fetched")
                
        except Exception as e:
            logger.error(f"❌ {symbol}: Failed with error: {e}")
        
        logger.info("-"*60)
    
    logger.info("\n" + "="*60)
    logger.info("✅ Download complete!")
    logger.info("="*60)
    
    # Show summary
    import glob
    parquet_files = glob.glob("data/raw/*.parquet")
    logger.info(f"\nTotal files downloaded: {len(parquet_files)}")
    for file in sorted(parquet_files):
        size_mb = os.path.getsize(file) / 1024 / 1024
        logger.info(f"  {os.path.basename(file):50} {size_mb:>8.2f} MB")
    
    logger.info("\n🎯 Next steps:")
    logger.info("   1. Run: python scripts/prepare_training_data.py")
    logger.info("   2. Start training: python scripts/train_yup250.py --config config/training.yaml")


if __name__ == "__main__":
    download_historical_data()

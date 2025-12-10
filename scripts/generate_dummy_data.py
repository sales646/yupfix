"""
Generate dummy parquet data for YUP-250 training
Creates realistic-looking OHLCV data for all symbols
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def generate_dummy_ohlcv(symbol: str, n_samples: int = 100000, freq='1s'):
    """
    Generate dummy OHLCV data with all required columns.
    
    Args:
        symbol: Symbol name
        n_samples: Number of samples to generate
        freq: Frequency (default 1s)
    """
    # Start from 30 days ago
    start_date = datetime.now() - timedelta(days=30)
    dates = pd.date_range(start_date, periods=n_samples, freq=freq)
    
    # Base price depending on symbol
    base_prices = {
        'EURUSD': 1.10,
        'GBPUSD': 1.27,
        'USDJPY': 150.0,
        'XAUUSD': 2000.0,
        'NAS100': 16000.0,
        'US30': 36000.0
    }
    base_price = base_prices.get(symbol, 100.0)
    
    # Generate realistic price movement
    np.random.seed(hash(symbol) % 2**32)
    returns = np.random.randn(n_samples) * 0.0001  # 0.01% std per second
    price = base_price * np.exp(returns.cumsum())
    
    # OHLCV
    df = pd.DataFrame({
        'open': price,
        'high': price * (1 + np.abs(np.random.randn(n_samples) * 0.0002)),
        'low': price * (1 - np.abs(np.random.randn(n_samples) * 0.0002)),
        'close': price,
        'volume': np.random.randint(1000, 10000, n_samples),
    }, index=dates)
    
    # Additional required columns
    df['buy_volume'] = (df['volume'] * (0.5 + np.random.randn(n_samples) * 0.1)).clip(0)
    df['sell_volume'] = df['volume'] - df['buy_volume']
    
    # Bid/Ask
    spread = base_price * 0.00001  # 1 pip
    df['bid_close'] = df['close'] - spread / 2
    df['ask_close'] = df['close'] + spread / 2
    
    # Spread stats
    df['spread_avg'] = spread * (1 + np.random.randn(n_samples) * 0.1).clip(0.5, 1.5)
    df['spread_max'] = df['spread_avg'] * (1.5 + np.random.rand(n_samples) * 0.5)
    
    # Tick count
    df['tick_count'] = np.random.randint(10, 100, n_samples)
    
    return df


def main():
    """Generate all parquet files"""
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'NAS100', 'US30']
    
    # Create directories
    train_dir = Path('data/train')
    val_dir = Path('data/val')
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating dummy data...")
    
    for symbol in symbols:
        print(f"  {symbol}...", end=' ')
        
        # Training data (100k samples = ~27 hours @ 1sec)
        train_df = generate_dummy_ohlcv(symbol, n_samples=100000)
        train_path = train_dir / f"{symbol}.parquet"
        train_df.to_parquet(train_path)
        print(f"train: {len(train_df)} rows", end=', ')
        
        # Validation data (20k samples = ~5.5 hours)
        val_df = generate_dummy_ohlcv(symbol, n_samples=20000)
        val_path = val_dir / f"{symbol}.parquet"
        val_df.to_parquet(val_path)
        print(f"val: {len(val_df)} rows ✓")
    
    print("\n✅ All data generated!")
    print(f"\nTrain: {train_dir.absolute()}")
    print(f"Val:   {val_dir.absolute()}")
    
    # Show example
    print("\nExample data structure:")
    sample_df = pd.read_parquet(train_dir / 'EURUSD.parquet')
    print(sample_df.head())
    print(f"\nColumns: {list(sample_df.columns)}")
    print(f"Shape: {sample_df.shape}")
    print(f"Date range: {sample_df.index[0]} to {sample_df.index[-1]}")


if __name__ == '__main__':
    main()

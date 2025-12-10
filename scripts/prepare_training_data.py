"""
Prepare downloaded parquet data for training
- Resample to 1-second bars (or keep 1-minute if that's what we use)
- Split into train/val sets
- Add required columns for microstructure features
"""
import pandas as pd
import numpy as np
from pathlib import Path

def prepare_data(input_path: str, symbol: str, train_ratio: float = 0.8):
    """
    Prepare downloaded data for training.
    
    Args:
        input_path: Path to raw parquet file
        symbol: Symbol name
        train_ratio: Ratio for train/val split
    """
    print(f"\nProcessing {symbol}...")
    
    # Load data
    df = pd.read_parquet(input_path)
    print(f"  Loaded: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    # Ensure we have required columns
    required_base = ['open', 'high', 'low', 'close', 'volume']
    
    # Check what we have
    missing = [col for col in required_base if col not in df.columns]
    if missing:
        print(f"  ⚠️  Missing columns: {missing}")
        return
    
    # Add microstructure columns if missing
    if 'buy_volume' not in df.columns:
        # Estimate buy/sell volume (assuming 50/50 with some randomness)
        df['buy_volume'] = (df['volume'] * (0.5 + np.random.randn(len(df)) * 0.05)).clip(0)
        df['sell_volume'] = df['volume'] - df['buy_volume']
        print("  Added buy_volume, sell_volume (estimated)")
    
    if 'bid_close' not in df.columns:
        # Estimate bid/ask from close with typical spread
        # Forex: ~1-2 pips, Indices: ~0.1%
        if symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
            spread_pct = 0.00001  # ~1 pip
        elif symbol in ['XAUUSD']:
            spread_pct = 0.0001   # ~$0.20
        else:  # Indices
            spread_pct = 0.0001   # ~0.01%
        
        spread = df['close'] * spread_pct
        df['bid_close'] = df['close'] - spread / 2
        df['ask_close'] = df['close'] + spread / 2
        print("  Added bid_close, ask_close (estimated)")
    
    if 'spread_avg' not in df.columns:
        spread = df['ask_close'] - df['bid_close']
        df['spread_avg'] = spread
        df['spread_max'] = spread * (1.5 + np.random.rand(len(df)) * 0.5)
        print("  Added spread_avg, spread_max")
    
    if 'tick_count' not in df.columns:
        # Estimate tick count (more volume = more ticks)
        df['tick_count'] = (df['volume'] / df['volume'].mean() * 50).clip(10, 200).astype(int)
        print("  Added tick_count (estimated)")
    
    # Split into train/val
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"  Train: {len(train_df)} rows ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"  Val:   {len(val_df)} rows ({val_df.index[0]} to {val_df.index[-1]})")
    
    # Save
    train_dir = Path('data/train')
    val_dir = Path('data/val')
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = train_dir / f"{symbol}.parquet"
    val_path = val_dir / f"{symbol}.parquet"
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    print(f"  ✓ Saved to {train_path}")
    print(f"  ✓ Saved to {val_path}")
    
    return train_df, val_df


def main():
    """Process all downloaded data"""
    raw_dir = Path('data/raw')
    
    # Find all parquet files
    parquet_files = list(raw_dir.glob('*.parquet'))
    
    if not parquet_files:
        print("No parquet files found in data/raw/")
        return
    
    print(f"Found {len(parquet_files)} files:")
    for f in parquet_files:
        print(f"  - {f.name}")
    
    # Process each file
    for file_path in parquet_files:
        # Extract symbol from filename (e.g., "EURUSD_1m_2023-11-23_2025-11-22.parquet" -> "EURUSD")
        symbol = file_path.stem.split('_')[0]
        
        try:
            prepare_data(str(file_path), symbol, train_ratio=0.8)
        except Exception as e:
            print(f"  ✗ Error processing {symbol}: {e}")
    
    print("\n" + "="*60)
    print("✅ Data preparation complete!")
    print("="*60)
    
    # Show summary
    train_dir = Path('data/train')
    val_dir = Path('data/val')
    
    train_files = list(train_dir.glob('*.parquet'))
    val_files = list(val_dir.glob('*.parquet'))
    
    print(f"\nTrain files ({len(train_files)}):")
    for f in sorted(train_files):
        df = pd.read_parquet(f)
        print(f"  {f.name:40} {len(df):>8} rows")
    
    print(f"\nVal files ({len(val_files)}):")
    for f in sorted(val_files):
        df = pd.read_parquet(f)
        print(f"  {f.name:40} {len(df):>8} rows")
    
    print("\n🎯 Ready for training!")
    print("   Run: python scripts/train_yup250.py --config config/training.yaml --val-path data/val")


if __name__ == '__main__':
    main()

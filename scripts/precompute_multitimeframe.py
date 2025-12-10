import polars as pl
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def calculate_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate 24 technical features for the given dataframe.
    Expects columns: open, high, low, close, volume, tick_count
    """
    # 1-5. Price & Volume (already present, ensuring names)
    # 6. Log Return
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return")
    )

    # 7. High-Low Range
    df = df.with_columns(
        (pl.col("high") - pl.col("low")).alias("hl_range")
    )

    # 8. ATR(14)
    # TR = max(H-L, abs(H-prev_C), abs(L-prev_C))
    df = df.with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs()
        ).alias("tr")
    )
    # ATR is typically RMA (Running Moving Average) or SMA. Using SMA for simplicity/speed unless RMA needed.
    # Standard ATR uses Wilder's Smoothing (RMA). RMA(x, n) = (prev_RMA * (n-1) + x) / n
    # Polars doesn't have built-in ewm_mean with 'adjust=False' exactly like pandas for RMA easily without recursion or specific ewm.
    # We'll use EWM with alpha=1/n which is equivalent to RMA.
    df = df.with_columns(
        pl.col("tr").ewm_mean(alpha=1/14, adjust=False, min_samples=14).alias("atr_14")
    ).drop("tr")

    # 9-12. SMA/EMA (20, 50)
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
        pl.col("close").rolling_mean(window_size=50).alias("sma_50"),
        pl.col("close").ewm_mean(span=20, adjust=False).alias("ema_20"),
        pl.col("close").ewm_mean(span=50, adjust=False).alias("ema_50"),
    ])

    # 13-14. MACD (12, 26, 9)
    ema_12 = pl.col("close").ewm_mean(span=12, adjust=False)
    ema_26 = pl.col("close").ewm_mean(span=26, adjust=False)
    macd_line = ema_12 - ema_26
    
    df = df.with_columns(macd_line.alias("macd_line"))
    df = df.with_columns(
        pl.col("macd_line").ewm_mean(span=9, adjust=False).alias("macd_signal")
    )

    # 15. RSI(14)
    delta = pl.col("close").diff()
    up = delta.clip(lower_bound=0)
    down = delta.clip(upper_bound=0).abs()
    
    # RSI uses RMA (Wilder's smoothing) usually
    avg_up = up.ewm_mean(alpha=1/14, adjust=False, min_samples=14)
    avg_down = down.ewm_mean(alpha=1/14, adjust=False, min_samples=14)
    
    rs = avg_up / avg_down
    rsi = 100 - (100 / (1 + rs))
    
    df = df.with_columns(rsi.alias("rsi_14"))

    # 16. Momentum(10) = Close - Close(10)
    df = df.with_columns(
        (pl.col("close") - pl.col("close").shift(10)).alias("momentum_10")
    )

    # 17. ROC(10) = (Close - Close(10)) / Close(10)
    df = df.with_columns(
        ((pl.col("close") - pl.col("close").shift(10)) / pl.col("close").shift(10)).alias("roc_10")
    )

    # 18. CCI(14)
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3
    sma_tp = tp.rolling_mean(window_size=14)
    mean_dev = (tp - sma_tp).abs().rolling_mean(window_size=14)
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    
    df = df.with_columns(cci.alias("cci_14"))

    # 19-21. Bollinger Bands (20, 2)
    sma_20 = pl.col("close").rolling_mean(window_size=20)
    std_20 = pl.col("close").rolling_std(window_size=20)
    
    df = df.with_columns([
        (sma_20 + 2 * std_20).alias("bb_upper"),
        sma_20.alias("bb_middle"),
        (sma_20 - 2 * std_20).alias("bb_lower"),
    ])

    # 22. VWAP (Rolling 1-day approx or just rolling window)
    # Since we don't have guaranteed day breaks, we'll use a rolling 1-day (assuming 1h bars = 24, 5m = 288, 1m = 1440)
    # But this varies by timeframe. Let's use a standard rolling window of 'period' size roughly equal to a day, or just a fixed window.
    # Better: Session VWAP requires resetting at 00:00.
    # Let's implement a simple Session VWAP if possible, else Rolling VWAP(20).
    # Given the request for "features", Rolling VWAP is often more robust for ML than resetting VWAP which has discontinuities.
    # We'll use Rolling VWAP (window=20) to be consistent with other indicators, or maybe larger.
    # Let's use a Rolling VWAP of 50 periods.
    vwap = (pl.col("close") * pl.col("volume")).rolling_sum(window_size=50) / pl.col("volume").rolling_sum(window_size=50)
    df = df.with_columns(vwap.alias("vwap"))

    # 23. Tick Count (already present)
    
    # 24. Spread (if available, else 0 or estimated)
    if "spread" not in df.columns:
        # If we have bid/ask, calc spread. Else 0.
        # Assuming input might not have it, we'll create a placeholder or estimate if needed.
        # For now, fill with 0 if missing to ensure 24 features exist.
        df = df.with_columns(pl.lit(0.0).alias("spread"))

    # Select exactly the 24 features + timestamp
    features = [
        "timestamp", 
        "open", "high", "low", "close", "volume",
        "log_return", "hl_range", "atr_14",
        "sma_20", "sma_50", "ema_20", "ema_50",
        "macd_line", "macd_signal",
        "rsi_14", "momentum_10", "roc_10", "cci_14",
        "bb_upper", "bb_middle", "bb_lower",
        "vwap", "tick_count", "spread"
    ]
    
    return df.select(features)

import argparse

def process_symbol(file_path: Path, output_base_dir: Path, limit: int = None):
    symbol = file_path.stem.split('_')[0]
    logger.info(f"Processing {symbol} from {file_path}")

    # Load data
    # Assuming parquet has 'timestamp' column or index. 
    # If it's 1s data, it might be large.
    try:
        q = pl.scan_parquet(file_path)
        
        # Check columns
        columns = q.collect_schema().names()
        # Ensure timestamp is present
        if "timestamp" not in columns:
            # Try to infer if it's the index (Polars doesn't have index, but maybe it's a column named 'time' or 'date')
            if "time" in columns:
                q = q.rename({"time": "timestamp"})
            elif "Date" in columns:
                q = q.rename({"Date": "timestamp"})
            # If strictly index in pandas parquet, it might be named '__index_level_0__' or similar if not preserved.
            # We'll assume standard 'timestamp' or 'time'.
        
        # Ensure lowercase columns
        # Re-fetch columns after potential rename (though rename is lazy, so schema might not update immediately without collect)
        # Actually, rename on LazyFrame returns a new LazyFrame.
        # But we need the current column names to map them.
        current_cols = q.collect_schema().names()
        q = q.rename({c: c.lower() for c in current_cols})
        
        # Sort by timestamp
        q = q.sort("timestamp")

        # Apply limit if specified (for testing)
        if limit:
            logger.info(f"  [TEST MODE] Limiting to first {limit} rows")
            q = q.limit(limit)

    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return

    # Timeframes to process
    timeframes = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "1hour": "1h"
    }

    for tf_name, tf_interval in timeframes.items():
        logger.info(f"  Resampling to {tf_name}...")
        
        # Aggregation
        # Polars dynamic groupby
        # We want to group by 'timestamp' with period 'tf_interval'
        # label='left' means 10:00 bucket covers 10:00 to 10:04:59 (for 5m)
        # We will then SHIFT the timestamp to the right.
        
        # Determine if we have OHLC or Tick data
        columns = q.collect_schema().names()
        has_ohlc = "open" in columns
        
        if not has_ohlc and "ask" in columns and "bid" in columns:
            # Tick data: Create price and volume
            q = q.with_columns([
                ((pl.col("ask") + pl.col("bid")) / 2).alias("price"),
                ((pl.col("ask_vol") + pl.col("bid_vol")) / 2).alias("volume")
            ])
            
            # Define aggregations for Tick -> OHLC
            aggs = [
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("volume").sum(),
                pl.col("volume").count().alias("tick_count"),
            ]
        else:
            # OHLC data: Resample
            aggs = [
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                pl.col("tick_count").sum() if "tick_count" in columns else pl.col("volume").count().alias("tick_count"),
            ]
            
        if "spread" in columns:
            aggs.append(pl.col("spread").mean())
        elif "ask" in columns and "bid" in columns:
             aggs.append((pl.col("ask") - pl.col("bid")).mean().alias("spread"))

        try:
            resampled = (
                q.group_by_dynamic("timestamp", every=tf_interval, period=tf_interval, closed="left", label="left")
                .agg(aggs)
                .collect() # Execute lazy query
            )

            # CRITICAL: SHIFT TIMESTAMP TO BAR-END
            # 1min -> +1m, 5min -> +5m, etc.
            # We can parse the interval string or just map it.
            offset_map = {
                "1m": timedelta(minutes=1),
                "5m": timedelta(minutes=5),
                "15m": timedelta(minutes=15),
                "1h": timedelta(hours=1),
            }
            offset = offset_map[tf_interval]
            
            resampled = resampled.with_columns(
                (pl.col("timestamp") + offset).alias("timestamp")
            )

            # Calculate Features
            features_df = calculate_features(resampled)
            
            # Drop initial NaNs caused by rolling windows (e.g. SMA50 needs 50 rows)
            # features_df = features_df.drop_nulls() # Optional: keep them or drop? Usually drop for training.
            # Let's keep them but maybe warn. Or just drop.
            # For "precompute", we usually want clean data.
            features_df = features_df.drop_nulls()

            # Save
            output_dir = output_base_dir / tf_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{symbol}.parquet"
            
            features_df.write_parquet(output_path)
            logger.info(f"    Saved {len(features_df)} rows to {output_path}")

        except Exception as e:
            logger.error(f"    Failed to process {tf_name} for {symbol}: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Precompute multi-timeframe features")
    parser.add_argument("--test", action="store_true", help="Run in test mode (limit rows)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows per file")
    args = parser.parse_args()

    limit = args.limit
    if args.test and limit is None:
        limit = 100000  # Default test limit

    raw_dir = Path("data/raw")
    output_base_dir = Path("data/features")
    
    # Find input files
    # Looking for *_tick_10y.parquet or similar. 
    # We'll search for any .parquet in data/raw that looks like a data file.
    files = list(raw_dir.glob("*.parquet"))
    
    if not files:
        logger.warning("No parquet files found in data/raw/")
        return

    logger.info(f"Found {len(files)} files to process.")
    if limit:
        logger.info(f"Test mode active: Limiting to {limit} rows per file.")
    
    for f in files:
        process_symbol(f, output_base_dir, limit=limit)

    logger.info("Done.")

if __name__ == "__main__":
    main()

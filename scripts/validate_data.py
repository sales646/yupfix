# scripts/validate_data.py
import pandas as pd
from pathlib import Path
import sys

REQUIRED_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'buy_volume', 'sell_volume', 
    'bid_close', 'ask_close',
    'spread_avg', 'spread_max', 'tick_count'
]

def validate_parquet(file_path: Path) -> dict:
    """Validate a single parquet file"""
    errors = []
    warnings = []
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        return {'file': file_path.name, 'errors': [f"Cannot read: {e}"], 'warnings': []}
    
    # 1. Check columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    
    # 2. Check data types
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns and not pd.api.types.is_float_dtype(df[col]):
            errors.append(f"{col} is not float: {df[col].dtype}")
    
    # 3. Check for NaN
    nan_pct = df.isna().sum() / len(df) * 100
    high_nan = nan_pct[nan_pct > 5]
    if len(high_nan) > 0:
        warnings.append(f"High NaN%: {high_nan.to_dict()}")
    
    # 4. Check for inf
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    for col in numeric_cols:
        if df[col].isin([float('inf'), float('-inf')]).any():
            errors.append(f"{col} contains inf values")
    
    # 5. Check price logic
    if all(c in df.columns for c in ['high', 'low', 'open', 'close']):
        bad_high = (df['high'] < df['low']).sum()
        if bad_high > 0:
            errors.append(f"{bad_high} rows where high < low")
        
        bad_range = ((df['open'] > df['high']) | (df['open'] < df['low'])).sum()
        if bad_range > 0:
            warnings.append(f"{bad_range} rows where open outside high-low range")
    
    # 6. Check sequence length
    if len(df) < 14400:
        warnings.append(f"Only {len(df)} rows, need 14400 for full sequence")
    
    # 7. Check timestamp monotonic
    if 'timestamp' in df.columns or df.index.name == 'timestamp':
        idx = df.index if df.index.name == 'timestamp' else df['timestamp']
        if not idx.is_monotonic_increasing:
            errors.append("Timestamps not monotonic increasing")
    
    return {
        'file': file_path.name,
        'rows': len(df),
        'columns': len(df.columns),
        'errors': errors,
        'warnings': warnings
    }


def validate_all_data(data_dir: str):
    """Validate all parquet files in directory"""
    data_path = Path(data_dir)
    results = []
    
    if not data_path.exists():
        print(f"❌ Directory not found: {data_path}")
        return False

    print(f"Scanning {data_path}...")
    files = list(data_path.glob("*.parquet"))
    if not files:
        print(f"❌ No parquet files found in {data_path}")
        return False

    for parquet_file in files:
        result = validate_parquet(parquet_file)
        results.append(result)
        
        status = "✅" if not result['errors'] else "❌"
        print(f"{status} {result['file']}: {result['rows']} rows")
        
        for err in result['errors']:
            print(f"   ❌ ERROR: {err}")
        for warn in result['warnings']:
            print(f"   ⚠️ WARNING: {warn}")
    
    # Summary
    total_errors = sum(len(r['errors']) for r in results)
    total_warnings = sum(len(r['warnings']) for r in results)
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {len(results)} files, {total_errors} errors, {total_warnings} warnings")
    
    if total_errors > 0:
        print("❌ DATA VALIDATION FAILED - Fix errors before training!")
        return False
    else:
        print("✅ DATA VALIDATION PASSED")
        return True


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    validate_all_data(data_dir)

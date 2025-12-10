# scripts/preflight_check.py
"""
Run this BEFORE training to catch all issues.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from src.yup250_pipeline import Yup250Pipeline
from src.targets.label_generator import create_labels_for_symbols, align_features_and_labels
from src.contracts.pipeline_validator import validate_pipeline


def main():
    print("="*60)
    print("🚀 YUP-250 PRE-FLIGHT CHECK")
    print("="*60)
    
    # 1. Load config
    print("\n[1] Loading config...")
    config_path = "config/training.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"   ✅ Config loaded: {config_path}")
    
    # 2. Check data exists
    print("\n[2] Checking data files...")
    data_path = Path(config['data']['train_path'])
    if not data_path.exists():
        print(f"   ❌ Data path not found: {data_path}")
        sys.exit(1)
    
    symbols = config['data']['symbols']
    missing = []
    for symbol in symbols:
        parquet_file = data_path / f"{symbol}.parquet"
        if not parquet_file.exists():
            missing.append(symbol)
        else:
            import pandas as pd
            df = pd.read_parquet(parquet_file)
            print(f"   ✅ {symbol}: {len(df):,} rows, {df.shape[1]} columns")
    
    if missing:
        print(f"   ❌ Missing data files: {missing}")
        sys.exit(1)
    
    # 3. Initialize pipeline
    print("\n[3] Initializing pipeline...")
    pipeline = Yup250Pipeline(config)
    print("   ✅ Pipeline initialized")
    
    # 4. Load data
    print("\n[4] Loading data...")
    data = pipeline.load_data(config['data']['train_path'])
    print(f"   ✅ Loaded {len(data)} symbols")
    
    # 5. Compute features
    print("\n[5] Computing features...")
    features = pipeline.prepare_features(data)
    for symbol, feat in features.items():
        print(f"   {symbol}: {feat.shape}")
    
    # 6. Generate labels
    print("\n[6] Generating labels...")
    labels = create_labels_for_symbols(data, horizon=12, delay=3)
    for symbol, lab in labels.items():
        print(f"   {symbol}: {lab.shape}")
    
    # 7. Align features and labels
    print("\n[7] Aligning features and labels...")
    features, labels = align_features_and_labels(features, labels)
    for symbol in features.keys():
        print(f"   {symbol}: features={len(features[symbol])}, labels={len(labels[symbol])}")
    
    # 8. Fit HMM
    print("\n[8] Fitting HMM...")
    pipeline.fit_hmm(data)
    print("   ✅ HMM fitted")
    
    # 9. Create model
    print("\n[9] Creating model...")
    model = pipeline.create_model(use_ensemble=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   ✅ Model on {device}")
    
    # 10. GPU Memory check
    print("\n[10] Checking GPU memory...")
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Total: {total_mem:.1f} GB")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Available: {total_mem - allocated:.1f} GB")
        
        # Estimate batch memory
        seq_len = config['data']['sequence_length']
        batch_size = config['training']['batch_size']
        d_input = config['model']['d_input']
        d_model = config['model']['d_model']
        n_layers = config['model']['n_layers']
        
        # Rough estimate: activations dominate
        activation_mb = batch_size * seq_len * d_model * n_layers * 4 / 1024**2
        print(f"   Estimated batch activations: ~{activation_mb:.0f} MB")
        
        if activation_mb > (total_mem - allocated) * 1024 * 0.8:
            print(f"   ⚠️ WARNING: May run out of memory!")
    else:
        print("   ⚠️ No GPU found, will train on CPU (slow)")
    
    # 11. Full pipeline validation
    print("\n[11] Running pipeline validation...")
    validate_pipeline(config, features, labels, model)
    
    # 12. Summary
    print("\n" + "="*60)
    print("✅ ALL PRE-FLIGHT CHECKS PASSED!")
    print("="*60)
    print("\nReady to train:")
    print(f"   python scripts/train_yup250.py --config {config_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

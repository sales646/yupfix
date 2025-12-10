#!/bin/bash
# YUP-250 H100 Setup Script
# Run this after uploading yupfix_code.zip and data files

echo "=========================================="
echo "  YUP-250 H100 Setup"
echo "=========================================="

# 1. Unzip code
echo "[1/5] Extracting code..."
unzip -o yupfix_code.zip -d yupfix
cd yupfix

# 2. Create virtual environment
echo "[2/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
echo "[3/5] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install polars pandas numpy scikit-learn hmmlearn pyyaml tensorboard gymnasium
pip install mamba-ssm  # Official Mamba with CUDA kernels

# 4. Create data directories
echo "[4/5] Setting up data directories..."
mkdir -p data/features/1min data/features/5min data/features/15min data/features/1hour
mkdir -p models/checkpoints_large logs/training_large

# 5. Instructions for data upload
echo "[5/5] Setup complete!"
echo ""
echo "=========================================="
echo "  NEXT STEPS:"
echo "=========================================="
echo ""
echo "1. Upload feature data to data/features/"
echo "   scp -r data/features/* user@h100:~/yupfix/data/features/"
echo ""
echo "2. Start training:"
echo "   python scripts/train_yup250.py --config config/training_large.yaml"
echo ""
echo "3. Monitor with TensorBoard:"
echo "   tensorboard --logdir logs/training_large"
echo ""
echo "=========================================="

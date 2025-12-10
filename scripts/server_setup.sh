#!/bin/bash
# Run this on the A100 server

echo "============================================"
echo "YUP-250 A100 Server Setup"
echo "============================================"

# 1. Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip
echo "[2/5] Upgrading pip..."
pip install --upgrade pip

# 3. Install PyTorch with CUDA
echo "[3/5] Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install requirements
echo "[4/5] Installing requirements..."
pip install -r requirements.txt

# 5. Install hmmlearn (should work on Linux)
echo "[5/5] Installing hmmlearn..."
pip install hmmlearn

# Verify GPU
echo ""
echo "============================================"
echo "Verifying GPU..."
echo "============================================"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "============================================"
echo "Setup complete! Run:"
echo "  python scripts/preflight_check.py"
echo "  python scripts/train_yup250.py --config config/training.yaml"
echo "============================================"

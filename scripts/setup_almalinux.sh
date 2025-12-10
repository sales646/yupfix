#!/bin/bash
# Setup script for FTMO Trading Bot on AlmaLinux 9
# Usage: ./scripts/setup_almalinux.sh

set -e

echo "==================================================="
echo "   FTMO Trading Bot - AlmaLinux Setup Assistant"
echo "==================================================="

# 1. System Update & Dependencies
echo "[1/4] Updating system and installing dependencies..."
sudo dnf update -y
sudo dnf install -y git python3.11 python3.11-devel python3.11-pip gcc make

# 2. Virtual Environment
echo "[2/4] Creating virtual environment (venv)..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
else
    python3.11 -m venv venv
    echo "Created 'venv'."
fi

# Activate venv for the script execution
source venv/bin/activate

# 3. Install Requirements
echo "[3/4] Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Final Instructions
echo "==================================================="
echo "   Setup Complete!"
echo "==================================================="
echo ""
echo "To run the bot:"
echo "  source venv/bin/activate"
echo "  python src/main.py"
echo ""
echo "IMPORTANT: MetaTrader 5 (MT5) requires Windows."
echo "You have two options for MT5:"
echo "  A) Run MT5 on a separate Windows VPS and connect via VPN/Tunnel."
echo "  B) Install Wine on this server to run MT5 (Advanced)."
echo ""
echo "See docs/linux_deployment.md for details."

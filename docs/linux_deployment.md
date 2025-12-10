# Linux Deployment Guide

This guide explains how to deploy the FTMO Trading Bot on a Linux server (e.g., AlmaLinux, Ubuntu).

## Architecture Overview

The system consists of two main parts:
1.  **The Brain (Python)**: Strategy logic, Risk Manager, Machine Learning. **(Runs natively on Linux)**
2.  **The Hands (MetaTrader 5)**: Execution terminal and data feed. **(Windows Native)**

Since MT5 is a Windows application, you cannot run it directly on Linux without a compatibility layer.

## Option A: Hybrid Deployment (Recommended)

Run the "Brain" on your Linux server and the "Hands" on a small Windows VPS or local PC.

### 1. Linux Server (The Brain)
*   **OS**: AlmaLinux 9 / Ubuntu 22.04
*   **Setup**: Run `scripts/setup_almalinux.sh`
*   **Config**: Update `config/config.yaml`:
    ```yaml
    bridge:
      host: "0.0.0.0" # Listen on all interfaces
    ```

### 2. Windows Machine (The Hands)
*   **OS**: Windows 10/11 or Server 2019+
*   **Software**: Install MetaTrader 5 and the ZeroMQ Bridge EA.
*   **Connection**:
    *   **VPN (Best)**: Install Tailscale or ZeroTier on both machines. Use the VPN IP in `config.yaml`.
    *   **SSH Tunnel**: Forward ports 5555/5556 from Windows to Linux.

## Option B: Wine Emulation (Advanced)

Run everything on one Linux server using Wine to emulate Windows for MT5.

### 1. Install Wine
On AlmaLinux 9:
```bash
sudo dnf config-manager --add-repo https://dl.winehq.org/wine-builds/almalinux/9/winehq.repo
sudo dnf install winehq-stable
```

### 2. Install MT5
1.  Download `mt5setup.exe` from your broker.
2.  Run with Wine: `wine mt5setup.exe`
3.  Follow the installer (headless servers may need Xvfb for GUI).

### 3. Run the Bot
1.  Start MT5 via Wine.
2.  Start Python Bot normally: `python src/main.py`
3.  Set `host: "localhost"` in `config.yaml`.

## Troubleshooting

*   **ZeroMQ Connection**: Ensure ports 5555 (REQ) and 5556 (SUB) are open in the firewall (`firewall-cmd --add-port=5555/tcp`).
*   **Headless MT5**: MT5 requires a GUI. On a headless server, use `xvfb-run` to create a virtual display.

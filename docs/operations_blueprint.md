# FTMO Trading Bot - Operations Blueprint

## 1. Infrastructure Architecture

### A. Training Server (The "Gym")
**Purpose**: Heavy computation, model training, hyperparameter optimization, and backtesting.
**Location**: Cloud (AWS/Lambda Labs) or Dedicated On-Prem Workstation.

*   **Hardware Specifications**:
    *   **GPU**: NVIDIA RTX 3090 / 4090 (24GB VRAM) or A100 (40GB/80GB). *Crucial for Transformer/LSTM training.*
    *   **CPU**: AMD Ryzen 9 7950X or Threadripper (High core count for parallel environment stepping).
    *   **RAM**: 64GB - 128GB DDR5 (To hold large tick datasets in memory).
    *   **Storage**: 2TB NVMe Gen4 SSD (Fast I/O for data loading).
*   **Software Stack**:
    *   **OS**: Ubuntu 22.04 LTS.
    *   **Environment**: Docker, PyTorch (CUDA 11.8+), Ray (Distributed Training), MLflow (Experiment Tracking).
    *   **Data**: HDF5/Parquet repository of Tick Data (2018-Present).

### B. Inference Node (The "Trader")
**Purpose**: Live execution, low latency, reliability.
**Location**: Local Dedicated PC or Low-Latency VPS (New York/London near broker servers).

*   **Hardware Specifications**:
    *   **CPU**: Intel Core i7/i9 (High single-core clock speed > 4.5GHz for main loop latency).
    *   **RAM**: 32GB.
    *   **Storage**: 512GB NVMe SSD.
    *   **Network**: Fiber connection (Ethernet, not WiFi), < 5ms ping to Broker Server.
*   **Software Stack**:
    *   **OS**: Windows 10/11 Pro (Required for MT5 Desktop).
    *   **Runtime**: Python 3.10 (Optimized), ONNX Runtime (for fast inference).
    *   **Execution**: MetaTrader 5 Terminal + ZMQ Bridge.

### C. Alternative: Hybrid Deployment (Linux VPS + Windows)
*Use this if your VPS provider only offers Linux.*

1.  **The Brain (Linux VPS)**:
    *   **OS**: **Ubuntu 22.04 LTS** (Best choice from your list).
    *   **Role**: Runs the Python Strategy Engine, Risk Manager, and ZMQ Server.
    *   **Pros**: Superior stability for Python/ML, lower cost.
2.  **The Hands (Windows Machine)**:
    *   **OS**: Windows 10/11 (Local PC or separate Windows VPS).
    *   **Role**: Runs MT5 Terminal + ZMQ Client (Bridge).
    *   **Connectivity**: Connects to the Linux VPS via **SSH Tunnel** or **VPN** (ZeroTier/Tailscale) to bridge the ZMQ ports securely.
    *   **Note**: This adds slight latency (10-50ms) but allows you to use a Linux VPS for the heavy lifting.

---

## 2. Workflow: Training to Live

### Step 1: Training & Versioning
1.  **Data Sync**: Sync latest tick data from Local/VPS to Training Server.
2.  **Training Loop**:
    *   Run `scripts/train_rl.py` with updated hyperparameters.
    *   Log metrics (Reward, Entropy, Value Loss) to **MLflow**.
3.  **Evaluation**:
    *   Run `scripts/monte_carlo_ftmo.py` on holdout data.
    *   **Gatekeeper**: If `Pass_Rate > 60%` AND `Max_DD < 4%`, proceed.
4.  **Export**:
    *   Convert PyTorch model to **ONNX** (`model_v1.2.onnx`).
    *   Generate `config_v1.2.json` (Input schema, normalization params).
    *   Tag release in Git: `release/v1.2`.

### Step 2: Deployment
1.  **Pull**: On Inference Node, pull latest release from Git.
2.  **Validation**: Run `scripts/dry_run.py` with the new model to verify it loads and outputs valid signals.
3.  **Hot Swap**: Update `config.yaml` to point to new model path. Restart Python Service (MT5 stays open).

---

## 3. Risk Control & Guardian Layer

### A. Internal Limits (The "Soft" Stops)
*Configured in `config.yaml` (Stricter than FTMO)*
*   **Daily Loss Limit**: **3.5%** (FTMO is 5%).
    *   *Action*: Close all positions, sleep until 23:05 Server Time.
*   **Max Total Loss**: **7.0%** (FTMO is 10%).
    *   *Action*: **KILL SWITCH**. Send "CRITICAL" alert. Manual intervention required.
*   **Max Leverage**: 10x (Dynamic scaling based on Volatility).

### B. Live Guardrails (Real-time)
*   **News Filter**:
    *   Block entry 5 mins before High Impact USD/EUR news.
    *   Force close profitable scalps 1 min before news.
*   **Spread Monitor**:
    *   If `Spread > 2.0 pips` (EURUSD), block entry.
    *   If `Spread > 5.0 pips`, treat as "Flash Event" -> Halt trading.
*   **Execution Watchdog**:
    *   If `Order_Fill_Time > 500ms`, log warning.
    *   If `Slippage > 1.0 pip`, pause trading for 5 mins.

### C. Global Kill Switches
1.  **Automated**:
    *   `Equity < Hard_Stop_Level`: Immediate `OrderCloseAll()`.
    *   `ZMQ_Heartbeat_Lost`: MT5 EA detects Python silence > 5s -> Closes all positions (Safety mechanism).
2.  **Manual**:
    *   "PANIC BUTTON" on Dashboard (Web UI) -> Sends ZMQ `CLOSE_ALL` command.
    *   Physical: Close MT5 Terminal (EA `OnDeinit` can be configured to close all).

---

## 4. Dashboard & Monitoring

### A. Tech Stack
*   **Backend**: FastAPI (exposes status endpoints).
*   **Frontend**: Streamlit (Python-based UI) or Grafana.
*   **Database**: SQLite (local logs) + Prometheus (Time series metrics).

### B. Dashboard Panels
1.  **System Health**:
    *   MT5 Connection Status (Green/Red).
    *   Latency (ms).
    *   Last Heartbeat timestamp.
2.  **FTMO Tracker**:
    *   Current Daily Equity Change (%).
    *   Distance to Daily Limit (%).
    *   Distance to Profit Target (%).
    *   Trading Days Count.
3.  **Active Trades**:
    *   Symbol, Type, Vol, PnL, Duration.
    *   "Close" button per trade.
4.  **Performance**:
    *   Equity Curve (Live).
    *   Win Rate (Today).

### C. Alerts (Telegram/Discord)
*   **INFO**: "Trade Opened: Buy EURUSD 1.0 lot".
*   **WARNING**: "Daily Drawdown > 2.0%".
*   **CRITICAL**: "Daily Limit Hit! Trading Disabled." / "ZMQ Disconnected!".

---

## 5. Operational Roadmap

### Phase 1: The Sandbox (2 Weeks)
*   **Account**: FTMO Free Trial (Demo).
*   **Goal**: Verify technical stability (0 crashes, 0 missed trades).
*   **Risk**: Minimal (0.5% risk per trade).

### Phase 2: The Challenge (30 Days)
*   **Account**: FTMO Challenge ($100k).
*   **Goal**: +10% Profit.
*   **Risk**: Standard (1.0% risk per trade).
*   **Ops**: Daily morning check of VPS. Weekly model re-training if regime shifts.

### Phase 3: Verification (60 Days)
*   **Account**: FTMO Verification.
*   **Goal**: +5% Profit.
*   **Risk**: Conservative (0.5% risk per trade). *Don't rush.*

### Phase 4: Funded & Scaling
*   **Account**: FTMO Funded Account.
*   **Goal**: Consistent Payouts.
*   **Rule**: Withdraw 80% of profit monthly. Leave 20% for buffer.
*   **Scaling**:
    *   Use `TradeCopier` to replicate trades to multiple funded accounts (max allocation $400k).
    *   Diversify: Run different model versions on different accounts (Ensemble at Account Level).

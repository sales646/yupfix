# FTMO Trading Bot - Project Summary

## 1. Project Overview
**Objective**: Develop a professional, institutional-grade trading bot optimized to pass the FTMO Challenge and manage funded accounts.
**Core Philosophy**: "Survival First, Profit Second." The system is engineered to strictly adhere to FTMO's drawdown limits (5% Daily / 10% Total) while generating consistent alpha.

## 2. System Architecture
*   **Hybrid Design**: 
    *   **Python (Brain)**: Handles complex logic, Machine Learning, and Strategy execution.
    *   **MetaTrader 5 (Hands)**: Handles market connectivity, data streaming, and order execution via a custom **ZeroMQ Bridge**.
*   **Guardian Layer**: A dedicated Risk Management module that acts as a proxy between the strategy and the market. It enforces internal limits (stricter than FTMO's) to prevent rule violations.

## 3. Implemented Strategies (The "Ensemble")
The bot utilizes a **Meta-Controller** that dynamically weights signals from multiple sub-strategies based on the detected market regime (Trending vs. Ranging).

| Component | Type | Role | Status |
| :--- | :--- | :--- | :--- |
| **Momentum** | Statistical | Captures medium-term trends with volatility scaling. | ✅ Complete |
| **Heston Model** | Quant | Stochastic volatility modeling for risk adjustment. | ✅ Complete |
| **XGBoost** | Supervised ML | Predicts short-term direction (Up/Down/Neutral) using tabular features. | ✅ Complete |
| **LSTM** | Deep Learning | Captures sequential patterns and long-term dependencies in price action. | ✅ Complete |
| **PPO Agent** | Reinforcement Learning | Optimizes risk-adjusted returns directly via trial-and-error in a simulated environment. | ✅ Complete |
| **Regime Detector** | HMM | Classifies market state to adjust strategy weights dynamically. | ✅ Complete |

## 4. System Hardening & Compliance
*   **Real-time FTMO Guard**: `RiskManager` tracks `start_of_day_equity` (reset at NY Close) to enforce the 5% Daily Loss limit in real-time.
*   **Payout Lock-in**: Automated logic to reduce risk or halt trading when Profit > 5% to secure payout eligibility.
*   **Shadow Mode**: Execution capability to run the full system on live data without sending real orders.
*   **Model Registry**: Versioning and metadata tracking for all ML models.

## 5. Real Data & Training Pipeline (New)
*   **Data Sources**: Integrated **Polygon.io** (5-Year Historical OHLCV) and **Finnhub** (Live News Events).
*   **Streaming Training**: Implemented a **"Download-While-Training"** pipeline (`scripts/train_xgb_stream.py`).
    *   **Background Downloader**: Fetches data in 30-day chunks in a separate thread.
    *   **Incremental Learning**: XGBoost model updates continuously as new data chunks arrive, eliminating wait times.
*   **News Filter**: Live blocking of trades during high-impact economic events (NFP, FOMC, etc.).

## 6. Operations & Monitoring
*   **Dashboard**: A real-time **Streamlit** application (`src/dashboard/app.py`) visualizing Equity, Active Trades, Risk Metrics, and Latency.
*   **Blueprint**: A comprehensive Operations Manual (`docs/operations_blueprint.md`) detailing hardware specs (GPU/VPS), deployment workflows, and emergency procedures.

## 7. Next Steps for the User
1.  **Deploy**: Set up the Python environment and MT5 terminal on a low-latency VPS.
2.  **Train**: Run `python scripts/train_xgb_stream.py` to start the 5-year procedural training process.
3.  **Verify**: Run `monte_carlo_ftmo.py` with the trained models to confirm a >60% Pass Rate.
4.  **Launch**: Start with Phase 1 (FTMO Free Trial) using the Dashboard to monitor performance.

---
*System Version: 1.2.0 (Real Data Integrated) | Status: Ready for Training & Deployment*

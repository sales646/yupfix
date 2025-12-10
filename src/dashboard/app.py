import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
import yaml

# Page Config
st.set_page_config(
    page_title="FTMO Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Config
@st.cache_data
def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Mock Data (Replace with Redis/ZMQ/File reads in real app)
def get_account_state():
    # In real app, read from shared state file or Redis
    return {
        "balance": 100000.0,
        "equity": 102500.0,
        "daily_start_equity": 100000.0,
        "open_pnl": 2500.0,
        "daily_dd_pct": -0.025, # Profit actually
        "total_dd_pct": -0.025
    }

def get_active_trades():
    return pd.DataFrame([
        {"Symbol": "EURUSD", "Type": "BUY", "Volume": 1.0, "OpenPrice": 1.1050, "CurrentPrice": 1.1075, "PnL": 250.0},
        {"Symbol": "NAS100", "Type": "SELL", "Volume": 0.5, "OpenPrice": 15000, "CurrentPrice": 14950, "PnL": 2250.0}
    ])

def get_logs():
    # Read last 10 lines of log file
    # Mocking for now
    return [
        "2023-10-27 10:00:01 [INFO] StrategyController: Signal Confirmed EURUSD: Action=BUY",
        "2023-10-27 10:00:02 [INFO] ExecutionHandler: Order Executed Successfully: Ticket 12345",
        "2023-10-27 10:05:00 [INFO] RiskManager: Daily Drawdown 0.00% OK"
    ]

# --- Sidebar ---
st.sidebar.title("🤖 FTMO Bot")
st.sidebar.markdown("---")
st.sidebar.header("System Status")
st.sidebar.success("System Online")
st.sidebar.info(f"Mode: {config['system']['mode']}")

if st.sidebar.button("🚨 PANIC: CLOSE ALL"):
    st.sidebar.error("Sending CLOSE ALL command...")
    # Send ZMQ Command here

# --- Main Layout ---
st.title("Command Center")

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
state = get_account_state()

with col1:
    st.metric("Equity", f"${state['equity']:,.2f}", f"{state['open_pnl']:,.2f}")
with col2:
    daily_limit = config['risk']['daily_loss_limit_pct']
    # Calculate distance to limit
    # Daily Loss is (Start - Equity) / Start
    # If Profit, DD is negative.
    # Let's show "Daily PnL %"
    daily_pnl_pct = (state['equity'] - state['daily_start_equity']) / state['daily_start_equity']
    st.metric("Daily PnL", f"{daily_pnl_pct:.2%}", delta_color="normal")
    st.progress(min(max((daily_pnl_pct + daily_limit) / daily_limit, 0.0), 1.0))
    st.caption(f"Limit: -{daily_limit:.1%}")

with col3:
    target = config['risk']['profit_target_pct']
    total_pnl_pct = (state['equity'] - 100000) / 100000
    st.metric("Total Return", f"{total_pnl_pct:.2%}")
    st.progress(min(max(total_pnl_pct / target, 0.0), 1.0))
    st.caption(f"Target: {target:.1%}")

with col4:
    st.metric("Active Trades", "2")
    st.metric("Trading Days", "4 / 10") # Mock

# --- Advanced Metrics ---
st.markdown("---")
st.subheader("Advanced Metrics")
m1, m2, m3 = st.columns(3)

with m1:
    st.info("Scaling Eligibility")
    st.write("Status: **Pending**")
    st.progress(0.25) # 1/4 months
    st.caption("10% Profit over 4 Months required")

with m2:
    st.warning("Execution Latency")
    st.metric("Avg Latency", "125 ms", "-5 ms")
    st.caption("Target: < 200 ms")

with m3:
    st.error("News Filter")
    st.write("Next Event: **NFP (USD)**")
    st.write("Time: **14:30**")
    st.caption("Trading will be blocked 5 mins prior")

# Charts & Tables
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Equity Curve vs Volatility")
    # Mock Chart with Volatility
    chart_data = pd.DataFrame({
        'Equity': np.cumsum(np.random.randn(50)) + 100000,
        'Volatility': np.random.rand(50) * 100
    })
    st.line_chart(chart_data)

    st.subheader("Active Positions")
    st.dataframe(get_active_trades(), use_container_width=True)

with col_right:
    st.subheader("Live Logs")
    logs = get_logs()
    for log in logs:
        st.text(log)
        
    st.subheader("Risk Monitor")
    st.write("Guardian Layer: **ACTIVE**")
    st.write(f"Max Drawdown Buffer: {config['risk']['max_drawdown_buffer']:.2%}")
    
# Auto-refresh
time.sleep(1)
st.rerun()

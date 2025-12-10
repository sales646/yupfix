import vectorbt as vbt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def generate_data(n=10000):
    np.random.seed(42)
    price = 100 * np.exp(np.cumsum(np.random.normal(0, 0.001, n)))
    return pd.Series(price, name='Close')

def run_backtest():
    print("Generating Data...")
    close = generate_data(50000) # 1-min data for ~2 months
    
    # 1. Momentum Signal
    lookback = 20
    momentum = close.pct_change(lookback)
    
    # 2. Volatility Scaling
    vol = close.pct_change().rolling(20).std() * np.sqrt(252 * 1440) # Annualized
    target_vol = 0.15
    leverage = target_vol / vol
    leverage = leverage.fillna(0).replace(np.inf, 0)
    leverage = np.clip(leverage, 0, 5.0) # Cap at 5x
    
    # 3. Generate Entries/Exits
    entries = (momentum > 0) & (momentum.shift(1) <= 0)
    exits = (momentum < 0) & (momentum.shift(1) >= 0)
    
    # 4. Run Portfolio Simulation
    print("Running Simulation...")
    portfolio = vbt.Portfolio.from_signals(
        close, 
        entries, 
        exits, 
        size=leverage,
        size_type='targetpercent', # Rebalance to target leverage
        init_cash=100000,
        freq='1min'
    )
    
    # 5. Metrics
    print("\n--- Backtest Results ---")
    print(f"Total Return: {portfolio.total_return():.2%}")
    print(f"Sharpe Ratio: {portfolio.sharpe_ratio():.2f}")
    print(f"Max Drawdown: {portfolio.max_drawdown():.2%}")
    print(f"Win Rate: {portfolio.win_rate():.2%}")
    
    # FTMO Checks
    daily_dd = portfolio.daily_returns().min()
    print(f"Max Daily Loss (approx): {daily_dd:.2%}")
    
    # Plot
    # portfolio.plot().show() # Uncomment to show plot

if __name__ == "__main__":
    run_backtest()

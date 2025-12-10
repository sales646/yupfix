import pandas as pd
import numpy as np
from src.models.rl.ppo_agent import PPOAgent

def generate_synthetic_data_with_features(n=10000):
    """Generate random walk data with required features."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, n)
    price = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': price,
        'high': price * 1.001,
        'low': price * 0.999,
        'close': price,
        'volume': np.random.randint(100, 1000, n)
    })
    
    # Add Features expected by Env
    df['ret_1'] = df['close'].pct_change(1).fillna(0)
    df['ret_5'] = df['close'].pct_change(5).fillna(0)
    df['vol_20'] = df['ret_1'].rolling(20).std().fillna(0)
    df['ma_20'] = df['close'].rolling(20).mean()
    df['dist_ma_20'] = ((df['close'] - df['ma_20']) / df['ma_20']).fillna(0)
    
    # Mock RSI
    df['rsi'] = 50 + np.random.normal(0, 10, n)
    
    return df.dropna()

def main():
    print("Generating synthetic data...")
    df = generate_synthetic_data_with_features()
    
    print("Initializing PPO Agent...")
    agent = PPOAgent()
    
    print("Training...")
    agent.train(df, total_timesteps=5000)
    
    print("Training Complete. Model Saved.")

if __name__ == "__main__":
    main()

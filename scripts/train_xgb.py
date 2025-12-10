import pandas as pd
import numpy as np
from src.models.supervised.xgb_model import XGBoostModel

def generate_synthetic_data(n=10000):
    """Generate random walk data for testing."""
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
    return df

def main():
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    print("Initializing Model...")
    model = XGBoostModel()
    
    print("Training...")
    model.train(df)
    
    print("Testing Prediction...")
    probs = model.predict(df.tail(50))
    print(f"Probabilities: {probs}")

if __name__ == "__main__":
    main()

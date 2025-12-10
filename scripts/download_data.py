import sys
import os
import yaml
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.providers import PolygonLoader
from src.data.storage import DataStorage

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    api_key = config['api_keys']['polygon']
    
    if api_key == "YOUR_POLYGON_API_KEY":
        print("Error: Please set your Polygon API Key in config/config.yaml")
        return

    loader = PolygonLoader(api_key)
    storage = DataStorage()
    
    symbols = config['trading']['symbols']
    
    # Download last 5 years
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")
    
    print(f"Downloading data from {start_date} to {end_date}...")
    
    for symbol in symbols:
        # Polygon Ticker Mapping
        poly_symbol = symbol
        if symbol == "NAS100": poly_symbol = "I:NDX"
        elif symbol == "US30": poly_symbol = "I:DJI"
        elif symbol == "SPX500": poly_symbol = "I:SPX"
        elif len(symbol) == 6: # Forex assumption
            poly_symbol = f"C:{symbol}"
            
        print(f"Fetching {symbol} (Polygon: {poly_symbol})...")
        df = loader.fetch_data(poly_symbol, start_date, end_date)
        
        if df is not None:
            # Save to storage
            # DataStorage expects a specific format, let's just save raw parquet for now
            # or adapt to storage.save_candles if implemented
            
            filename = f"data/raw/{symbol}_1m_{start_date}_{end_date}.parquet"
            os.makedirs("data/raw", exist_ok=True)
            df.to_parquet(filename)
            print(f"Saved to {filename}")
        else:
            print(f"Failed to fetch {symbol}")

if __name__ == "__main__":
    main()

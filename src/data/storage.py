import pandas as pd
import os
from datetime import datetime
from typing import Dict

class DataStorage:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def save_candle(self, symbol: str, candle: Dict):
        """Append candle to daily parquet file."""
        df = pd.DataFrame([candle])
        
        # Partition by Year/Month
        date_str = candle['timestamp'].strftime('%Y-%m-%d')
        year = candle['timestamp'].year
        
        path = os.path.join(self.data_dir, symbol, str(year))
        if not os.path.exists(path):
            os.makedirs(path)
            
        file_path = os.path.join(path, f"{date_str}.parquet")
        
        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['timestamp'])
            combined_df.to_parquet(file_path)
        else:
            df.to_parquet(file_path)

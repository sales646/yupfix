import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import logging
from typing import Optional

class PolygonLoader:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        self.logger = logging.getLogger("PolygonLoader")

    def fetch_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = '1', timespan: str = 'minute') -> Optional[pd.DataFrame]:
        """
        Fetches historical aggregates (candles) from Polygon.io.
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'C:EURUSD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Multiplier (e.g., 1)
            timespan: 'minute', 'hour', 'day'
        """
        # Format for Polygon: /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}
        url = f"{self.base_url}/{symbol}/range/{timeframe}/{timespan}/{start_date}/{end_date}"
        
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,
            'apiKey': self.api_key
        }
        
        self.logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}...")
        
        try:
            retries = 0
            max_retries = 5
            
            while retries < max_retries:
                response = requests.get(url, params=params)
                
                if response.status_code == 429:
                    wait_time = 2 ** retries # Exponential backoff: 1, 2, 4, 8, 16s
                    self.logger.warning(f"Rate limit hit (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    retries += 1
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if data['status'] not in ['OK', 'DELAYED'] or not data.get('results'):
                    self.logger.warning(f"No data found or error: {data}")
                    return None
                    
                results = data['results']
                df = pd.DataFrame(results)
                
                # Rename columns to standard format
                df = df.rename(columns={
                    't': 'timestamp',
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume',
                    'n': 'trades'
                })
                
                # Convert timestamp (ms) to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                self.logger.info(f"Successfully fetched {len(df)} rows.")
                return df
                
            self.logger.error("Max retries exceeded for rate limit.")
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return None

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

class DataIngestion:
    def __init__(self, zmq_client, history_size: int = 1000):
        self.logger = logging.getLogger("DataIngestion")
        self.zmq_client = zmq_client
        self.tick_buffer = defaultdict(list)
        self.current_candles = {} # Symbol -> Candle
        self.history = defaultdict(list) # Symbol -> List[Candle]
        self.history_size = history_size

    def process_tick(self, tick_data: Dict):
        """Process incoming tick data from ZMQ."""
        symbol = tick_data['symbol']
        price = (tick_data['bid'] + tick_data['ask']) / 2 # Mid price
        timestamp = tick_data['time'] # Milliseconds
        
        # Aggregate into 1-minute candles
        self._update_candle(symbol, price, timestamp)

    def _update_candle(self, symbol: str, price: float, timestamp: int):
        # Convert ms to minute timestamp
        dt = datetime.fromtimestamp(timestamp / 1000.0)
        minute_start = dt.replace(second=0, microsecond=0)
        
        if symbol not in self.current_candles:
            self.current_candles[symbol] = {
                'timestamp': minute_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 0
            }
        
        candle = self.current_candles[symbol]
        
        # Check if new minute started
        if minute_start > candle['timestamp']:
            self._finalize_candle(symbol, candle)
            # Start new candle
            self.current_candles[symbol] = {
                'timestamp': minute_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 0
            }
            candle = self.current_candles[symbol]
            
        # Update current candle
        candle['high'] = max(candle['high'], price)
        candle['low'] = min(candle['low'], price)
        candle['close'] = price
        candle['volume'] += 1 # Tick volume

    def _finalize_candle(self, symbol: str, candle: Dict):
        """Push completed candle to storage/strategy."""
        self.logger.info(f"New Candle {symbol}: {candle}")
        
        # Update History
        self.history[symbol].append(candle)
        if len(self.history[symbol]) > self.history_size:
            self.history[symbol].pop(0)
            
        # Here we would push to a Queue or Database
        # storage.save_candle(symbol, candle)

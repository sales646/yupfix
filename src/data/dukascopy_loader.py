import requests
import pandas as pd
import struct
import lzma
import os
from datetime import datetime, timedelta
from typing import Optional
import logging

class DukascopyLoader:
    """
    Fetches historical tick data from Dukascopy's free data feed with CSV caching.
    """
    def __init__(self):
        self.base_url = "https://datafeed.dukascopy.com/datafeed"
        self.logger = logging.getLogger("DukascopyLoader")
        
    def _symbol_to_dukascopy_format(self, symbol: str) -> str:
        """Convert standard symbol to Dukascopy format."""
        if symbol in ['NAS100', 'US30', 'SPX500']:
            self.logger.warning(f"{symbol} not available on Dukascopy (Forex only). Skipping.")
            return None
        return symbol
    
    def _decode_bi5(self, data: bytes) -> pd.DataFrame:
        """Decode Dukascopy .bi5 binary format."""
        try:
            if not data or len(data) < 10:
                return pd.DataFrame()
            
            try:
                decompressed = lzma.decompress(data)
            except lzma.LZMAError:
                return pd.DataFrame()
            
            tick_size = 20
            num_ticks = len(decompressed) // tick_size
            
            if num_ticks == 0:
                return pd.DataFrame()
            
            ticks = []
            for i in range(num_ticks):
                chunk = decompressed[i*tick_size:(i+1)*tick_size]
                if len(chunk) < tick_size:
                    continue
                    
                timestamp_ms, ask, bid, ask_vol, bid_vol = struct.unpack('>IIIff', chunk)
                point = 0.00001
                
                ticks.append({
                    'timestamp': timestamp_ms,
                    'ask': ask * point,
                    'bid': bid * point,
                    'ask_volume': ask_vol,
                    'bid_volume': bid_vol
                })
            
            return pd.DataFrame(ticks) if ticks else pd.DataFrame()
            
        except Exception as e:
            self.logger.debug(f"Error decoding bi5: {e}")
            return pd.DataFrame()
    
    def fetch_hour_wrapper(self, args):
        """Wrapper for parallel execution."""
        symbol, year, month, day, hour = args
        duka_symbol = self._symbol_to_dukascopy_format(symbol)
        
        url = f"{self.base_url}/{duka_symbol}/{year}/{month-1:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                df = self._decode_bi5(response.content)
                
                if not df.empty:
                    base_time = datetime(year, month, day, hour)
                    df['timestamp'] = base_time + pd.to_timedelta(df['timestamp'], unit='ms')
                    df['mid'] = (df['ask'] + df['bid']) / 2
                    return df
        except Exception:
            pass
            
        return None

    def fetch_day(self, symbol: str, year: int, month: int, day: int) -> Optional[pd.DataFrame]:
        """Fetch one full day of tick data (24 hours) in PARALLEL."""
        import concurrent.futures
        
        duka_symbol = self._symbol_to_dukascopy_format(symbol)
        if not duka_symbol:
            return None
        
        args_list = [(symbol, year, month, day, h) for h in range(24)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(self.fetch_hour_wrapper, args_list))
            
        day_ticks = [df for df in results if df is not None]
        
        if not day_ticks:
            return None
        
        self.logger.info(f"{duka_symbol} {year}-{month:02d}-{day:02d}: {len(day_ticks)}/24 hours fetched")
        return pd.concat(day_ticks, ignore_index=True)
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str, resample='1min') -> Optional[pd.DataFrame]:
        """Fetch historical data with CSV caching."""
        # Check cache first
        cache_dir = "data/cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/{symbol}_{start_date}_{end_date}_{resample}.csv"
        
        if os.path.exists(cache_file):
            self.logger.info(f"✅ Loading {symbol} from cache: {cache_file}")
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                self.logger.info(f"✅ Loaded {len(df)} cached candles")
                return df
            except Exception as e:
                self.logger.warning(f"Cache read failed: {e}, re-downloading")
        
        # Download if not cached
        symbol_duka = self._symbol_to_dukascopy_format(symbol)
        if symbol_duka is None:
            return None
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_ticks = []
        current = start
        
        while current <= end:
            day_ticks = self.fetch_day(symbol_duka, current.year, current.month, current.day)
            if day_ticks is not None and not day_ticks.empty:
                all_ticks.append(day_ticks)
            
            current += timedelta(days=1)
        
        if not all_ticks:
            self.logger.warning(f"No data found for {symbol} from {start_date} to {end_date}")
            return None
        
        # Combine all ticks
        df = pd.concat(all_ticks, ignore_index=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Resample to OHLC
        ohlc = df['mid'].resample(resample).ohlc()
        volume = df['ask_volume'].resample(resample).sum()
        
        result = pd.DataFrame({
            'open': ohlc['open'],
            'high': ohlc['high'],
            'low': ohlc['low'],
            'close': ohlc['close'],
            'volume': volume
        })
        
        result.dropna(inplace=True)
        
        # Save to cache
        try:
            result.to_csv(cache_file)
            self.logger.info(f"💾 Cached {len(result)} candles to {cache_file}")
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")
        
        self.logger.info(f"Fetched {len(result)} candles for {symbol}")
        return result

import threading
import queue
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import logging
from src.data.dukascopy_loader import DukascopyLoader

class BackgroundDownloader:
    def __init__(self, symbols: List[str], start_date: datetime, end_date: datetime, chunk_size_days: int = 30):
        self.logger = logging.getLogger("BackgroundDownloader")
        self.loader = DukascopyLoader()
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.chunk_size = timedelta(days=chunk_size_days)
        
        self.data_queue = queue.Queue()
        self.is_running = False
        self.is_complete = False
        self.thread = None
        
    def start(self):
        """Starts the background download thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self._download_worker, daemon=True)
        self.thread.start()
        self.logger.info("Background download started.")

    def _download_worker(self):
        current_date = self.start_date
        
        while current_date < self.end_date and self.is_running:
            next_date = min(current_date + self.chunk_size, self.end_date)
            
            start_str = current_date.strftime("%Y-%m-%d")
            end_str = next_date.strftime("%Y-%m-%d")
            
            self.logger.info(f"Downloading chunk: {start_str} to {end_str}")
            
            chunk_data = {}
            for symbol in self.symbols:
                # Dukascopy only has Forex, skip indices
                if symbol in ['NAS100', 'US30', 'SPX500']:
                    self.logger.warning(f"Skipping {symbol} (not available on Dukascopy)")
                    continue
                
                df = self.loader.fetch_data(symbol, start_str, end_str)
                if df is not None and not df.empty:
                    chunk_data[symbol] = df
            
            if chunk_data:
                self.data_queue.put(chunk_data)
                self.logger.info(f"Chunk {start_str}-{end_str} added to queue.")
            else:
                self.logger.warning(f"No data for chunk {start_str}-{end_str}")
                
            current_date = next_date + timedelta(days=1) # Next day
            
        self.is_complete = True
        self.is_running = False
        self.logger.info("Background download complete.")

    def get_next_chunk(self, timeout: int = 5) -> Optional[dict]:
        """
        Retrieves the next data chunk from the queue.
        Returns None if queue is empty and download is complete.
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            if self.is_complete:
                return None
            return "WAIT" # Indicate waiting for data

    def wait_for_initial_load(self, min_chunks: int = 1):
        """Blocks until a minimum number of chunks are available."""
        self.logger.info("Waiting for initial data load...")
        while self.data_queue.qsize() < min_chunks and not self.is_complete:
            time.sleep(1)
        self.logger.info("Initial load complete. Starting training.")

```python
import logging
from typing import Dict
from src.bridge.zmq_client import ZmqClient
from src.guardian.risk_manager import RiskManager

class ExecutionHandler:
    def __init__(self, zmq_client: ZmqClient, risk_manager: RiskManager, shadow_mode: bool = False):
        self.logger = logging.getLogger("ExecutionHandler")
        self.zmq_client = zmq_client
        self.risk_manager = risk_manager
        self.shadow_mode = shadow_mode

import time

    def execute_signal(self, signal: Dict):
        """
        Executes a trading signal if allowed by Risk Manager.
        """
        start_time = time.time()
        self.logger.info(f"Processing Signal: {signal}")
        
        # 1. Risk Check
        if not self.risk_manager.check_trade_allowed(signal):
            self.logger.warning("Signal Rejected by Risk Manager")
            return
            
        # 2. Execution
        symbol = signal['symbol']
        action = signal['action']
        volume = signal['volume']
        sl = signal['sl']
        tp = signal['tp']
        
        if self.shadow_mode:
            self.logger.info(f"[SHADOW] Would execute: {action} {symbol} {volume} lots @ Market")
            return {'status': 'shadow_filled', 'ticket': -1}
        
        try:
            if action == 'BUY':
                response = self.zmq_client.open_order(symbol, 'buy', volume, sl, tp)
            elif action == 'SELL':
                response = self.zmq_client.open_order(symbol, 'sell', volume, sl, tp)
            else:
                return
                
            latency_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Order Executed: {response} | Latency: {latency_ms:.2f}ms")
            self.risk_manager.record_trade(signal) 
            
        except Exception as e:
            self.logger.error(f"Execution Failed: {e}")
```

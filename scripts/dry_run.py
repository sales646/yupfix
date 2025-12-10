import logging
import time
import pandas as pd
import numpy as np
from src.strategies.controller import StrategyController
from src.guardian.risk_manager import RiskManager
from src.execution.handler import ExecutionHandler
from src.utils.logger import setup_logger

# Mock ZMQ Client
class MockZmqClient:
    def open_order(self, symbol, order_type, volume, sl, tp):
        print(f"[MOCK] Sending Order: {symbol} {order_type} {volume} lots")
        return {'status': 'ok', 'ticket': 12345}

def main():
    logger = setup_logger("DryRun", level="INFO")
    logger.info("Starting Dry Run Verification...")
    
    # Config
    config = {
        'risk': {
            'daily_loss_limit_pct': 0.05,
            'max_loss_limit_pct': 0.10,
            'profit_target_pct': 0.10,
            'max_drawdown_buffer': 0.005
        }
    }
    
    # Initialize Components
    strategy_controller = StrategyController(config)
    risk_manager = RiskManager(config)
    zmq_client = MockZmqClient()
    execution_handler = ExecutionHandler(zmq_client, risk_manager)
    
    # Simulate Data Stream
    logger.info("Simulating Data Stream...")
    history = {'EURUSD': []}
    
    for i in range(60): # Run for 60 ticks/minutes
        # Generate random candle
        price = 1.1000 + np.random.normal(0, 0.0005)
        candle = {
            'timestamp': time.time(),
            'open': price, 'high': price+0.0001, 'low': price-0.0001, 'close': price,
            'volume': 100
        }
        history['EURUSD'].append(candle)
        
        # Run Strategy Logic
        signal = strategy_controller.calculate_signal(history)
        
        if signal:
            logger.info(f"Signal Generated: {signal}")
            execution_handler.execute_signal(signal)
        else:
            # logger.debug("No Signal")
            pass
            
        time.sleep(0.1) # Fast forward
        
    logger.info("Dry Run Complete. System Integration Verified.")

if __name__ == "__main__":
    main()

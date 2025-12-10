import time
import yaml
import sys
import os
from src.utils.logger import setup_logger
from src.bridge.zmq_client import ZmqClient
from src.guardian.risk_manager import RiskManager

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Setup Logging
    logger = setup_logger("FTMO_Bot", level="INFO")
    logger.info("Starting FTMO Trading Bot...")

    # Load Config
    try:
        config = load_config("config/config.yaml")
        logger.info("Configuration loaded.")
    except Exception as e:
        logger.critical(f"Failed to load config: {e}")
        sys.exit(1)

    # Initialize Components
    zmq_client = ZmqClient(
        req_port=config['bridge']['zmq_req_port'],
        sub_port=config['bridge']['zmq_sub_port'],
        host=config['bridge']['host']
    )
    
    risk_manager = RiskManager(config)
    
    from src.data.ingestion import DataIngestion
    from src.execution.handler import ExecutionHandler
    from src.strategies.controller import StrategyController
    
    data_ingestion = DataIngestion(zmq_client)
    execution_handler = ExecutionHandler(zmq_client, risk_manager)
    strategy_controller = StrategyController(config)
    
    # Main Loop
    logger.info("Entering Main Loop...")
    try:
        while True:
            # 1. Check Connection
            # if not zmq_client.check_connection():
            #     logger.warning("MT5 Bridge disconnected. Retrying...")
            #     time.sleep(5)
            #     continue
            
            # 2. Process Incoming Ticks (Mocking for now, real impl would use ZMQ SUB loop)
            # try:
            #     tick = zmq_client.sub_socket.recv_json(flags=zmq.NOBLOCK)
            #     data_ingestion.process_tick(tick)
            # except zmq.Again:
            #     pass
            
            # 3. Update Account State
            # account_info = zmq_client.get_account_info()
            # risk_manager.update_account_state(account_info)
            
            # 4. Check Hard Stops
            # risk_manager.check_hard_stop()
            
            # 5. Strategy Logic
            # Pass history to strategy controller
            signal = strategy_controller.calculate_signal(data_ingestion.history)
            
            if signal:
                execution_handler.execute_signal(signal)
            
            time.sleep(0.01) # Fast loop
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")

if __name__ == "__main__":
    main()

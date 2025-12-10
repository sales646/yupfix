import zmq
import json
import logging
import time
from typing import Dict, Any, Optional

class ZmqClient:
    def __init__(self, req_port: int = 5555, sub_port: int = 5556, host: str = "localhost"):
        self.context = zmq.Context()
        
        # REQ Socket for sending commands (Order, GetHistory, etc.)
        self.req_socket = self.context.socket(zmq.REQ)
        self.req_socket.connect(f"tcp://{host}:{req_port}")
        
        # SUB Socket for receiving streaming data (Ticks)
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{host}:{sub_port}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ZMQ Client initialized. REQ: {req_port}, SUB: {sub_port}")

    def send_command(self, command: str, params: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Send a command to MT5 and wait for response."""
        message = {
            "command": command,
            "params": params,
            "timestamp": time.time()
        }
        
        try:
            self.req_socket.send_string(json.dumps(message))
            response = self.req_socket.recv_string()
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error sending command {command}: {e}")
            return {"status": "error", "message": str(e)}

    def get_account_info(self) -> Dict[str, Any]:
        return self.send_command("GET_ACCOUNT_INFO")

    def get_history(self, symbol: str, timeframe: str, count: int) -> Dict[str, Any]:
        return self.send_command("GET_HISTORY", {"symbol": symbol, "timeframe": timeframe, "count": count})

    def open_order(self, symbol: str, order_type: str, volume: float, sl: float = 0, tp: float = 0) -> Dict[str, Any]:
        return self.send_command("OPEN_ORDER", {
            "symbol": symbol, 
            "type": order_type, 
            "volume": volume, 
            "sl": sl, 
            "tp": tp
        })

    def check_connection(self) -> bool:
        try:
            response = self.send_command("PING")
            return response.get("status") == "pong"
        except:
            return False

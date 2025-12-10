import unittest
from unittest.mock import MagicMock
import logging
from src.strategies.controller import StrategyController
from src.guardian.risk_manager import RiskManager
from src.execution.handler import ExecutionHandler

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Config
        self.config = {
            'risk': {
                'daily_loss_limit_pct': 0.05,
                'max_loss_limit_pct': 0.10,
                'profit_target_pct': 0.10,
                'max_drawdown_buffer': 0.005,
                'payout_lock_enabled': True,
                'scaling_tracker_enabled': True
            },
            'system': {
                'shadow_mode': False
            }
        }
        
        # Mocks
        self.zmq_client = MagicMock()
        self.zmq_client.open_order.return_value = {'status': 'ok', 'ticket': 12345}
        
        # Components
        self.risk_manager = RiskManager(self.config)
        self.execution_handler = ExecutionHandler(self.zmq_client, self.risk_manager, shadow_mode=False)
        
        # Strategy Controller (Mocking internal strategies to force signal)
        self.strategy_controller = StrategyController(self.config)
        self.strategy_controller.momentum = MagicMock()
        self.strategy_controller.momentum.calculate_signal.return_value = {
            'action': 'BUY', 'leverage': 1.0
        }
        # Disable other strategies for simple test
        self.strategy_controller.use_ml = False
        self.strategy_controller.use_rl = False
        self.strategy_controller.use_regime = False

    def test_full_flow_execution(self):
        """Test Data -> Strategy -> Risk -> Execution (Success)"""
        history = {'EURUSD': [{'close': 1.1000} for _ in range(60)]}
        
        # 1. Generate Signal
        signal = self.strategy_controller.calculate_signal(history)
        self.assertIsNotNone(signal)
        self.assertEqual(signal['action'], 'BUY')
        
        # 2. Execute
        self.execution_handler.execute_signal(signal)
        
        # 3. Verify ZMQ called
        self.zmq_client.open_order.assert_called_once()
        args = self.zmq_client.open_order.call_args
        self.assertEqual(args[0][0], 'EURUSD')
        self.assertEqual(args[0][1], 'buy')

    def test_risk_block_daily_loss(self):
        """Test Risk Manager blocking trade due to Daily Loss"""
        # Simulate Daily Loss
        self.risk_manager.start_of_day_equity = 100000
        self.risk_manager.update_account_state(94000, 94000) # 6% Loss
        
        signal = {'symbol': 'EURUSD', 'action': 'BUY', 'volume': 1.0, 'sl': 0, 'tp': 0}
        
        # Execute
        self.execution_handler.execute_signal(signal)
        
        # Verify ZMQ NOT called
        self.zmq_client.open_order.assert_not_called()

    def test_shadow_mode(self):
        """Test Shadow Mode does not send real orders"""
        self.execution_handler.shadow_mode = True
        
        signal = {'symbol': 'EURUSD', 'action': 'BUY', 'volume': 1.0, 'sl': 0, 'tp': 0}
        self.execution_handler.execute_signal(signal)
        
        # Verify ZMQ NOT called
        self.zmq_client.open_order.assert_not_called()

if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL) # Silence logs during test
    unittest.main()

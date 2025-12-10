"""
Live Runner Skeleton
Orchestrates the entire trading system in a simulated live loop.
Connects Data -> Validation -> Features -> Model -> RL -> Risk -> Execution.
"""
import time
import logging
import pandas as pd
import torch
import yaml
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.validator import DataValidator
from src.features.microstructure import MicrostructureFeatureEngineer
from src.risk.manager import RiskManager
from src.execution.execution_model import ExecutionModel
from src.contracts.config import RiskConfig
# from src.models.mamba_full import MambaTrader # Import your model class
# from src.rl.ppo_agent import PPOAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LiveRunner")

class LiveTradingSystem:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.symbols = self.config['data']['symbols']
        
        # 1. Components
        self.validator = DataValidator(self.config.get('validation', {}))
        self.feature_engineer = MicrostructureFeatureEngineer(normalize=True)
        self.risk_manager = RiskManager(self.config.get('risk', {}))
        self.execution_model = ExecutionModel() # In real live, this is an API wrapper
        
        # 2. Load Models (Mocked for skeleton)
        logger.info("Loading Models...")
        self.model = self._load_mamba_model()
        self.agent = self._load_rl_agent()
        
        # 3. State
        self.positions = {s: 0.0 for s in self.symbols}
        self.equity = 100000.0
        self.is_running = False
        
    def _load_config(self, path):
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)
        if 'risk' in raw:
            RiskConfig(**raw['risk'])
        return raw
            
    def _load_mamba_model(self):
        # Load pytorch model
        # model = MambaTrader(...)
        # model.load_state_dict(...)
        return "MambaModel"
        
    def _load_rl_agent(self):
        # Load PPO agent
        return "PPOAgent"
        
    def fetch_live_data(self):
        """
        Mock fetching latest 1-min bar for all symbols.
        In production: Connect to MT5/IB/Binance API.
        """
        data = {}
        now = pd.Timestamp.now()
        for s in self.symbols:
            # Create dummy dataframe with required columns
            df = pd.DataFrame({
                'open': [1.1000], 'high': [1.1005], 'low': [1.0995], 'close': [1.1002],
                'volume': [1000], 'buy_volume': [600], 'sell_volume': [400],
                'tick_count': [50], 'spread_avg': [0.0001], 'spread_max': [0.0002]
            }, index=[now])
            data[s] = df
        return data
        
    def run_cycle(self):
        """Main Logic Loop (runs every minute)"""
        logger.info("--- Starting Cycle ---")
        
        # 1. Fetch Data
        raw_data = self.fetch_live_data()
        
        # 2. Validate & Prepare
        valid_data = {}
        for sym, df in raw_data.items():
            is_valid, errors = self.validator.validate(df, sym)
            if is_valid:
                # 3. Feature Engineering
                # Note: We need history for rolling features! 
                # In real live, we maintain a buffer of last N bars.
                # Here we assume df contains enough history or we handle it.
                # Here we assume df contains enough history or we handle it.
                feats_list = self.feature_engineer.compute_features(df)
                # Convert to DF for model input
                feats_df = pd.DataFrame([f.dict() for f in feats_list])
                valid_data[sym] = feats_df
            else:
                logger.error(f"Skipping {sym} due to validation errors: {errors}")
                
        if not valid_data:
            logger.warning("No valid data this cycle.")
            return
            
        # 4. Model Inference (Mamba)
        # preds = self.model(valid_data)
        logger.info("Mamba Inference...")
        
        # 5. RL Decision (PPO)
        # action = self.agent.get_action(preds)
        logger.info("RL Agent Decision...")
        target_positions = {s: 0.1 for s in self.symbols} # Dummy Buy 0.1 lots
        
        # 6. Risk Check (Guardian)
        safe_positions = self.risk_manager.check_correlation_exposure(target_positions)
        
        # 7. Execution
        for sym, target in safe_positions.items():
            current = self.positions[sym]
            if target != current:
                # Execute trade
                # price = self.execution_model.execute(...)
                logger.info(f"EXECUTE {sym}: {current} -> {target}")
                self.positions[sym] = target
                
        logger.info("--- Cycle Complete ---")

    def start(self):
        self.is_running = True
        logger.info("System Started. Press Ctrl+C to stop.")
        try:
            while self.is_running:
                self.run_cycle()
                time.sleep(5) # Run every 5 seconds for demo (usually 60s)
        except KeyboardInterrupt:
            logger.info("Stopping...")
            self.is_running = False

if __name__ == "__main__":
    # Create dummy config if not exists
    config_path = "config/live_config.yaml"
    if not Path(config_path).exists():
        with open(config_path, 'w') as f:
            yaml.dump({
                'data': {'symbols': ['EURUSD', 'GBPUSD']},
                'risk': {'max_drawdown': 0.05},
                'validation': {'max_gap_seconds': 300}
            }, f)
            
    system = LiveTradingSystem(config_path)
    system.start()

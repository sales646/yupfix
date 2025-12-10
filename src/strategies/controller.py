import logging
import pandas as pd
from typing import Dict, List
from src.strategies.statistical.momentum import MomentumStrategy
from src.strategies.statistical.heston import HestonModel
from src.models.supervised.xgb_model import XGBoostModel
from src.models.supervised.lstm_model import LSTMModel
from src.models.rl.ppo_agent import PPOAgent
from src.strategies.regime_detector import RegimeDetector
import numpy as np

class StrategyController:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger("StrategyController")
        self.config = config
        
        # 1. Momentum
        self.momentum = MomentumStrategy(
            lookback=20, 
            vol_target=config['risk']['daily_loss_limit_pct'] * 2 
        )
        
        # 2. Heston (Volatility)
        self.heston = HestonModel()
        
        # 3. XGBoost (Supervised - Tabular)
        self.xgb_model = XGBoostModel()
        try:
            self.xgb_model.load()
            self.use_xgb = True
        except:
            self.use_xgb = False
            
        # 4. LSTM (Supervised - Sequence)
        self.lstm_model = LSTMModel()
        try:
            self.lstm_model.load()
            self.use_lstm = True
        except:
            self.use_lstm = False
            
        # 5. RL Agent (PPO)
        self.ppo_agent = PPOAgent()
        try:
            self.ppo_agent.load()
            self.use_rl = True
        except:
            self.use_rl = False
            
        # 6. Regime Detector
        self.regime_detector = RegimeDetector()
        try:
            self.regime_detector.load()
            self.use_regime = True
        except:
            self.use_regime = False

    def calculate_signal(self, history: Dict[str, List[Dict]]) -> Dict:
        """
        Aggregates signals from all strategies using Regime-based Weighting.
        """
        final_signal = None
        
        for symbol, candles_list in history.items():
            if len(candles_list) < 60: continue
            
            df = pd.DataFrame(candles_list)
            
            # --- 1. Detect Regime ---
            regime = 0 
            if self.use_regime:
                regime = self.regime_detector.predict(df)
            
            # --- 2. Get Sub-Signals ---
            
            # Momentum
            mom_result = self.momentum.calculate_signal(df)
            
            # Heston Volatility Check
            # If Heston Vol > Threshold, reduce size or block
            # For now, just log it
            # self.heston.calibrate(df['close'].pct_change()) # Too slow for live loop?
            # heston_vol = self.heston.get_volatility_forecast()
            
            # ML Signals
            xgb_probs = {'up': 0, 'down': 0}
            if self.use_xgb:
                xgb_probs = self.xgb_model.predict(df)
                
            lstm_probs = {'up': 0, 'down': 0}
            if self.use_lstm:
                lstm_probs = self.lstm_model.predict(df)
            
            # RL Action
            rl_action = 0
            if self.use_rl:
                obs = np.array([0, 0, 0, 50, 0, 0]) 
                rl_action = self.ppo_agent.predict(obs)
            
            # --- 3. Ensemble Logic (Meta-Controller) ---
            
            # Base Weights
            w_mom = 0.4
            w_xgb = 0.3
            w_lstm = 0.3
            
            # Dynamic Adjustment based on Regime
            if regime == 1: # Trending
                w_mom = 0.6
                w_xgb = 0.2
                w_lstm = 0.2
            elif regime == 0: # Ranging
                w_mom = 0.2
                w_xgb = 0.4
                w_lstm = 0.4
            
            # Voting
            score = 0
            if mom_result['action'] == 'BUY': score += w_mom
            elif mom_result['action'] == 'SELL': score -= w_mom
            
            score += (xgb_probs['up'] - xgb_probs['down']) * w_xgb
            score += (lstm_probs['up'] - lstm_probs['down']) * w_lstm
            
            # Final Decision
            action = 'HOLD'
            if score > 0.3: action = 'BUY'
            elif score < -0.3: action = 'SELL'
            
            # RL Override
            if self.use_rl and rl_action != 0:
                if rl_action == 1: action = 'BUY'
                elif rl_action == 2: action = 'SELL'
            
            if action != 'HOLD':
                self.logger.info(f"Ensemble Signal {symbol}: Action={action}, Score={score:.2f}, Regime={regime}")
                
                final_signal = {
                    'symbol': symbol,
                    'action': action,
                    'volume': 1.0 * mom_result['leverage'], 
                    'sl': 0, 
                    'tp': 0
                }
                break
                
        return final_signal

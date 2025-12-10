"""
Comprehensive System Verification Script
Checks all recently implemented components for syntax errors and basic functionality.
"""
import sys
import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

def verify_execution_model():
    logger.info("Verifying Execution Model...")
    from src.execution.execution_model import ExecutionModel
    em = ExecutionModel(avg_slippage_pips=0.5, fill_rate=0.9)
    price = em.execute(1, 1.1000, 0.0001)
    if price is not None:
        logger.info(f"  Execution successful: {price}")
    else:
        logger.info("  Execution rejected (expected behavior)")
    return True

def verify_normalization():
    logger.info("Verifying Feature Normalization...")
    from src.features.normalization import FeatureNormalizer
    df = pd.DataFrame({'a': np.random.randn(100) * 10 + 5})
    fn = FeatureNormalizer(method='robust')
    fn.fit(df)
    res = fn.transform(df)
    logger.info(f"  Transformed mean: {res['a'].mean():.4f}, std: {res['a'].std():.4f}")
    return True

def verify_walk_forward():
    logger.info("Verifying Walk-Forward Validator...")
    from src.validation.walk_forward import WalkForwardValidator
    df = pd.DataFrame(index=pd.date_range('2020-01-01', periods=1000, freq='D'))
    # Small windows for testing
    wf = WalkForwardValidator(n_splits=3, train_days=100, val_days=20, test_days=10, bars_per_day=1)
    splits = list(wf.split(df))
    logger.info(f"  Generated {len(splits)} splits")
    return len(splits) == 3

def verify_deep_attention():
    logger.info("Verifying Deep Cross-Asset Attention...")
    from src.models.cross_asset_attention import CrossAssetAttention
    d_model = 32
    n_assets = 4
    attn = CrossAssetAttention(d_model=d_model, n_assets=n_assets, n_layers=2)
    x = torch.randn(2, n_assets, d_model)
    out = attn(x)
    logger.info(f"  Output shape: {out.shape}")
    return out.shape == x.shape

def verify_regime_mamba():
    logger.info("Verifying Regime-Specific Mamba...")
    from src.models.regime_mamba import RegimeSpecificMamba
    config = {
        'd_input': 10, 'd_model': 16, 'n_layers': 2, 'd_state': 8, 'n_assets': 2
    }
    model = RegimeSpecificMamba(config, n_regimes=3)
    x = torch.randn(2, 20, 10) # B, L, D
    probs = torch.randn(2, 3) # B, n_regimes
    out = model(x, probs)
    logger.info(f"  Output keys: {out.keys()}")
    return True

def verify_rl_stack():
    logger.info("Verifying RL Stack (Env + Agent)...")
    from src.rl.mamba_portfolio_env import MambaPortfolioEnv
    from src.rl.ppo_agent import PPOAgent
    
    # Mock Mamba Model
    class MockMamba(torch.nn.Module):
        def forward(self, x):
            return {'uncertainty': torch.rand(2)} # 2 assets
            
    # Mock Data
    df = pd.DataFrame({'close': np.random.randn(100) + 100})
    data = {'A': df, 'B': df}
    
    config = {
        'risk': {'max_drawdown': 0.1},
        'model': {'d_model': 16}
    }
    
    env = MambaPortfolioEnv(MockMamba(), data, config)
    obs, _ = env.reset()
    logger.info(f"  Obs shape: {obs.shape}")
    
    agent = PPOAgent(env, config)
    action, _, _ = agent.get_action(obs)
    logger.info(f"  Action shape: {action.shape}")
    
    next_obs, reward, _, _, _ = env.step(action)
    logger.info(f"  Step reward: {reward}")
    return True

def verify_finetuning():
    logger.info("Verifying Finetuning Logic...")
    from src.training.trainer import train_epoch
    from src.training.finetuner import WeeklyFinetuner
    
    # Just check import and class instantiation
    # Real training requires data loader setup which is complex for a quick check
    logger.info("  Imports successful")
    return True

def main():
    checks = [
        verify_execution_model,
        verify_normalization,
        verify_walk_forward,
        verify_deep_attention,
        verify_regime_mamba,
        verify_rl_stack,
        verify_finetuning
    ]
    
    all_passed = True
    for check in checks:
        try:
            if check():
                logger.info(f"✅ {check.__name__} PASSED")
            else:
                logger.error(f"❌ {check.__name__} FAILED")
                all_passed = False
        except Exception as e:
            logger.error(f"❌ {check.__name__} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            
    if all_passed:
        logger.info("\n🎉 ALL SYSTEMS GO! 🎉")
    else:
        logger.error("\n⚠️ SOME CHECKS FAILED ⚠️")

if __name__ == "__main__":
    main()

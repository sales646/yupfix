import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict

class HestonModel:
    """
    Heston Stochastic Volatility Model.
    dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_1
    dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_2
    """
    def __init__(self):
        self.params = {
            'kappa': 2.0,   # Mean reversion speed
            'theta': 0.04,  # Long-run variance
            'xi': 0.3,      # Vol of Vol
            'rho': -0.7,    # Correlation between price and vol
            'v0': 0.04      # Initial variance
        }

    def calibrate(self, returns: pd.Series):
        """
        Calibrate parameters to historical data using Method of Moments or MLE.
        Simplified calibration for demonstration.
        """
        # 1. Estimate Variance
        vol = returns.rolling(20).std() * np.sqrt(252)
        var = vol ** 2
        
        # 2. Estimate Theta (Long run var)
        self.params['theta'] = var.mean()
        
        # 3. Estimate V0
        self.params['v0'] = var.iloc[-1]
        
        # 4. Estimate Kappa (Mean reversion) - Simplified
        # Regress dv on (theta - v)
        # For now, keep default or use simple heuristic
        self.params['kappa'] = 2.0 
        
        # 5. Estimate Xi (Vol of Vol)
        self.params['xi'] = var.std()
        
        # 6. Estimate Rho (Correlation)
        self.params['rho'] = returns.corr(var.diff())

    def simulate_path(self, S0: float, T: float, N: int, M: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic price paths using Heston dynamics.
        S0: Initial Price
        T: Time horizon (years)
        N: Number of time steps
        M: Number of paths
        """
        dt = T / N
        mu = 0.0 # Drift (assume neutral for stress test)
        
        # Parameters
        kappa = self.params['kappa']
        theta = self.params['theta']
        xi = self.params['xi']
        rho = self.params['rho']
        v0 = self.params['v0']
        
        # Arrays
        S = np.zeros((N + 1, M))
        v = np.zeros((N + 1, M))
        S[0] = S0
        v[0] = v0
        
        # Correlation matrix
        cov = np.array([[1, rho], [rho, 1]])
        
        for t in range(1, N + 1):
            # Generate correlated Brownian motions
            Z = np.random.multivariate_normal([0, 0], cov, M)
            W1 = Z[:, 0]
            W2 = Z[:, 1]
            
            # Update Variance (Full Truncation Euler)
            v_prev = v[t-1]
            v_prev_pos = np.maximum(v_prev, 0)
            dv = kappa * (theta - v_prev_pos) * dt + xi * np.sqrt(v_prev_pos * dt) * W2
            v[t] = v_prev + dv
            
            # Update Price
            dS = S[t-1] * mu * dt + S[t-1] * np.sqrt(v_prev_pos * dt) * W1
            S[t] = S[t-1] + dS
            
        return S, v

    def get_volatility_forecast(self) -> float:
        """Return current implied/forecasted volatility."""
        # In Heston, E[v_t] converges to theta.
        # Short term forecast is mix of v0 and theta.
        return np.sqrt(self.params['v0'])

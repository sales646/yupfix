import pandas as pd
import numpy as np
from arch import arch_model
from typing import Optional, Tuple

class GarchModel:
    def __init__(self, p: int = 1, q: int = 1, mean: str = 'Zero', vol: str = 'GARCH'):
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.model = None
        self.res = None

    def fit(self, returns: pd.Series) -> None:
        """
        Fit the GARCH model to the returns series.
        Returns should be scaled (e.g., * 100) for better convergence.
        """
        # Scale returns for numerical stability
        scaled_returns = returns * 100
        
        self.model = arch_model(scaled_returns, vol=self.vol, p=self.p, q=self.q, mean=self.mean)
        self.res = self.model.fit(disp='off')

    def forecast(self, horizon: int = 1) -> float:
        """
        Forecast volatility for the next step.
        Returns annualized volatility (decimal).
        """
        if self.res is None:
            raise ValueError("Model not fitted")
            
        forecasts = self.res.forecast(horizon=horizon)
        # Get next step variance
        var_forecast = forecasts.variance.iloc[-1, 0]
        
        # Convert back from scaled variance
        vol_forecast = np.sqrt(var_forecast) / 100
        
        return vol_forecast

    def get_conditional_volatility(self) -> pd.Series:
        """Return the fitted conditional volatility (decimal)."""
        if self.res is None:
            raise ValueError("Model not fitted")
        return self.res.conditional_volatility / 100

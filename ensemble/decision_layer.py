import numpy as np

class DecisionLayer:
    """
    Ensemble Decision Layer for FTMO Bot.
    Combines three signals:
    1. PPO (RL) – aggressive trader
    2. BC (Behavior Cloning) – conservative trader
    3. ML (XGBoost) – analyst signal

    The logic now respects all three components while avoiding hard vetoes that block activity.
    """

    def __init__(self, risk_manager=None):
        self.risk_manager = risk_manager  # Optional link to a global risk manager

    def _ml_action(self, ml_prob: float) -> int:
        """Convert ML probability to a discrete action.
        Returns 1 (Buy) if prob > 0.60, 2 (Sell) if prob < 0.40, otherwise 0 (Hold)."""
        if ml_prob > 0.60:
            return 1
        if ml_prob < 0.40:
            return 2
        return 0

    def get_decision(self, obs, rl_action: int, bc_action: int, ml_prob: float):
        """Return a final action based on a lightweight voting scheme.

        Parameters
        ----------
        obs : np.ndarray
            Current observation (unused in this simplified voting).
        rl_action : int
            Action from PPO (0=Hold, 1=Buy, 2=Sell).
        bc_action : int
            Action from BC (0=Hold, 1=Buy, 2=Sell).
        ml_prob : float
            Probability from the ML model (0.0‑1.0).

        Returns
        -------
        final_action : int
            Chosen action (0, 1, or 2).
        confidence : float
            Fraction of votes supporting the chosen action (0‑1).
        """
        # Gather votes – only include non‑hold signals from BC and ML
        votes = [rl_action]
        if bc_action != 0:
            votes.append(bc_action)
        ml_action = self._ml_action(ml_prob)
        if ml_action != 0:
            votes.append(ml_action)

        # Count occurrences of each possible action
        counts = {0: 0, 1: 0, 2: 0}
        for a in votes:
            counts[a] += 1

        # Choose the action with the highest count; ties fall back to PPO action
        max_count = max(counts.values())
        # Find actions that have this max count
        candidates = [a for a, c in counts.items() if c == max_count]
        if rl_action in candidates:
            final_action = rl_action
        else:
            # If PPO is not among the tied leaders, pick the first candidate (deterministic)
            final_action = candidates[0]

        confidence = max_count / len(votes)
        return final_action, confidence

    def calculate_position_size(self, account_balance, risk_per_trade_pct=0.01, sl_pips=10):
        """Calculate lot size based on risk percentage and stop‑loss.

        Parameters
        ----------
        account_balance : float
            Current account equity.
        risk_per_trade_pct : float, optional
            Fraction of the account to risk per trade (default 1%).
        sl_pips : int, optional
            Stop‑loss in pips (default 10).

        Returns
        -------
        float
            Rounded lot size.
        """
        risk_amount = account_balance * risk_per_trade_pct
        pip_value_per_lot = 10  # $10 per pip for 1 lot EURUSD
        lots = risk_amount / (sl_pips * pip_value_per_lot)
        return round(lots, 2)

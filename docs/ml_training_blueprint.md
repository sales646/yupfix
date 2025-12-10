# FTMO Bot - Machine Learning Training Blueprint

## 1. Model Architectures

### A. Supervised Learning (Signal Generators)
**Goal**: Predict directional probability and volatility regime.

#### 1. XGBoost / CatBoost (Tabular)
-   **Inputs**:
    -   Technical Indicators: RSI, MACD, Bollinger Bands, ATR, ADX.
    -   Lagged Returns: $r_{t-1}, r_{t-2}, ..., r_{t-5}$.
    -   Time Features: Hour of day, Day of week.
    -   Macro: Impact score of upcoming news.
-   **Targets**:
    -   **Direction**: Multiclass Classification `{0: Down, 1: Neutral, 2: Up}`.
        -   *Labeling*: `Up` if future return > 0.5 * ATR, `Down` if < -0.5 * ATR.
    -   **Volatility**: Regression of next 1-hour realized volatility.
-   **Loss Function**: `MultiClassLogLoss` (Direction), `RMSE` (Volatility).

#### 2. LSTM / Transformer (Sequence)
-   **Inputs**: Raw OHLCV sequences (window size 60).
-   **Architecture**:
    -   Embedding Layer -> Positional Encoding.
    -   2x Transformer Encoder Layers (Multi-head Attention).
    -   Global Average Pooling -> MLP Head.
-   **Target**: Same as XGBoost.

### B. Reinforcement Learning (Execution & Sizing)
**Goal**: Optimize risk-adjusted returns under FTMO constraints.

#### 1. PPO (Proximal Policy Optimization)
-   **Role**: Main trading agent.
-   **State Space**:
    -   Account State: Equity, DD%, Trades Open.
    -   Market State: Recent returns, Volatility forecast (GARCH).
    -   Signals: Output probabilities from Supervised Models.
-   **Action Space**: Discrete `{Buy, Sell, Close, Hold}` or Continuous `Position Size`.
-   **Reward Function**:
    $$ R_t = \text{LogReturn}_t - \lambda \cdot \text{Vol}_t - \text{Penalty}_{\text{DD}} $$

#### 2. Distributional RL (QR-DQN)
-   **Role**: Risk-aware agent for high-volatility regimes.
-   **Key Feature**: Predicts the *distribution* of Q-values (returns) rather than the mean. Allows avoiding actions with "fat left tails" (high risk of Max Loss).

### C. Meta-Controller (Ensemble Manager)
-   **Role**: Dynamically weight sub-models based on market regime.
-   **Logic**:
    -   If `Regime == Trending`: Weight(Momentum) > Weight(MeanRev).
    -   If `Regime == HighVol`: Reduce overall leverage, increase Weight(QR-DQN).

---

## 2. Training Strategy

### A. Curriculum Learning
1.  **Phase 1: Stability (The Sandbox)**
    -   Train on "easy" data (clear trends).
    -   Goal: Learn basic mechanics (Buy Low, Sell High).
    -   Constraints: None.
2.  **Phase 2: Noise & Volatility (The Real World)**
    -   Train on full historical dataset (2018-2023).
    -   Goal: Generalization.
3.  **Phase 3: FTMO Hard Mode (The Gauntlet)**
    -   Introduce hard penalties for Daily Loss (-5%) and Max Loss (-10%).
    -   Reward shaping: Bonus for hitting +10% profit target.

### B. Offline -> Online Pipeline
1.  **Behavior Cloning (BC)**: Pre-train PPO policy to mimic a profitable heuristic (e.g., Trend Following).
2.  **Offline RL**: Train on static dataset.
3.  **Online Fine-tuning**: Run in `FTMOEvaluator` simulator with live interaction.

---

## 3. Hyperparameter Optimization (Hyperopt)

**Tool**: Optuna (Bayesian Optimization).

### Search Space
-   **XGBoost**:
    -   `max_depth`: [3, 10]
    -   `learning_rate`: [0.001, 0.1]
    -   `subsample`: [0.5, 1.0]
-   **PPO**:
    -   `learning_rate`: [1e-5, 1e-3]
    -   `gamma`: [0.90, 0.999]
    -   `ent_coef`: [0.0, 0.1] (Exploration)

**Objective**: Maximize `Pass_Rate * Sortino_Ratio` over 100 Monte Carlo episodes.

---

## 4. Stress Testing & Robustness

### A. Simulation Artifacts
-   **Slippage**: Randomly add 0.1 - 2.0 pips to execution price.
-   **Latency**: Add 100ms - 500ms delay to order execution.
-   **Spread Expansion**: Multiply spread by 10x during news events.

### B. Monte Carlo Permutation
-   Shuffle the order of trades from a backtest.
-   Verify that < 1% of permutations hit Max Loss.

---

## 5. Model Export

-   **Format**: ONNX (Open Neural Network Exchange) for XGBoost/Sklearn. TorchScript for PyTorch.
-   **Schema**: JSON file defining expected input features and normalization parameters (mean, std).
-   **Versioning**: `models/v1.0.0/` containing `model.onnx`, `config.json`, `metrics.json`.

"""
Configuration file for cryptocurrency price prediction application.

This module contains all configuration constants used across the application.
Centralizing these values makes them easy to tune and document.
"""

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================

# Maximum number of samples to use for training/testing
# Rationale: Limits memory usage and training time while maintaining sufficient data
# for model convergence. 1000 days ≈ 2.7 years of daily data.
MAX_SAMPLES = 1000

# Train/test split ratio
# Rationale: 70/30 split is standard in ML. Provides enough training data while
# reserving sufficient test data for reliable evaluation.
TRAIN_RATIO = 0.70

# Random seed for reproducibility
# Rationale: Fixed seed ensures reproducible results across runs
RANDOM_SEED = 42

# Default trading interval
# Rationale: Daily data is most reliable for crypto, with good balance of
# granularity and availability
DEFAULT_INTERVAL = '1d'

# Interval to period mapping for yfinance
INT_TO_PERIODS = {
    '1m': '5d', '2m': '1mo', '5m': '1mo', '15m': '1mo', '30m': '1mo',
    '60m': '1mo', '90m': '1mo', '1h': '1y', '1d': '10y', '5d': '10y',
    '1wk': '10y', '1mo': '10y', '3mo': '10y'
}


# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# LSTM Model Configuration
LSTM_CONFIG = {
    'units': 10,           # Number of LSTM units - small network to prevent overfitting on crypto data
    'epochs': 100,         # Maximum training epochs - early stopping will halt if no improvement
    'batch_size': 32,      # Standard batch size for time series
    'patience': 30,        # Early stopping patience - allows time for convergence
    'verbose': False       # Suppress training output by default
}

# SARIMA Model Configuration
SARIMA_CONFIG = {
    'seasonal': True,      # Enable seasonal component for crypto data
    'm': 12,               # Seasonal period - monthly patterns in daily data
    'trace': False,        # Suppress auto_arima search output
    'start_p': 0,          # Minimum AR order
    'start_q': 0,          # Minimum MA order
    'max_p': 5,            # Maximum AR order
    'max_q': 5,            # Maximum MA order
    'd': None,             # Let auto_arima determine differencing order
    'D': None,             # Let auto_arima determine seasonal differencing
    'test': 'adf',         # Use Augmented Dickey-Fuller test for stationarity
    'error_action': 'ignore',
    'suppress_warnings': True,
    'stepwise': True       # Use stepwise search for efficiency
}

# GMDH Model Configurations
# Multiple GMDH variants for ensemble predictions
GMDH_CONFIGS = {
    'GMDH_1': {
        'type': 'Multi',
        'criterion_type': 'REGULARITY',  # Will be mapped to CriterionType enum
        'p_average': 1,
        'limit': 0.0,
        'k_best': 1,
        'polynomial_type': 'LINEAR'  # Will be mapped to PolynomialType enum
    },
    'GMDH_2': {
        'type': 'Ria',
        'criterion_type': 'REGULARITY',
        'p_average': 1,
        'limit': 0.0,
        'k_best': 3,
        'polynomial_type': 'QUADRATIC'
    }
}

# Transformer Model Configuration
TRANSFORMER_CONFIG = {
    'model_name': 'amazon/chronos-t5-tiny',
    'device_map': 'cpu',   # Use CPU for compatibility; change to 'mps' for Apple Silicon
    'torch_dtype': 'bfloat16',
    'num_samples': 3,      # Number of sample paths for prediction
    'temperature': 1.0,
    'top_k': 50,
    'top_p': 1.0
}


# =============================================================================
# PORTFOLIO OPTIMIZATION CONFIGURATION
# =============================================================================

# Correlation threshold for feature selection
# Rationale: Features with correlation > 0.9 are too similar and add little value
CORRELATION_THRESHOLD = 0.9

# Number of assets to consider for portfolio
# Rationale: 5-10 assets provide good diversification without over-complexity
DEFAULT_TOP_N_ASSETS = 5

# Validation/test split for portfolio evaluation
# Rationale: 50/50 split of test period for hyperparameter tuning vs final evaluation
PORTFOLIO_VAL_TEST_RATIO = 0.5

# Stop loss configuration (volatility-based sigma stop)
# Rationale: Crypto assets have heterogeneous volatility; a per-asset sigma stop
# adapts to each asset's actual risk profile rather than using a fixed percentage.
STOP_LOSS_SIGMA_MULTIPLIER = 1.5  # Stop placed 1.5σ below entry
STOP_LOSS_WINDOW = 20             # Rolling window (days) for σ estimation

# Signal significance thresholds
# A prediction must pass ALL applicable checks to be actionable (BUY/SELL).
# Failing any check results in HOLD and exclusion from portfolio optimization.
CLT_Z_SCORE = 1.96       # |pred| > CLT_Z_SCORE × residual_std  (~95% CI, all models)
VOLATILITY_Z_SCORE = 2.0 # |pred| / rolling_σ > VOLATILITY_Z_SCORE  (all models)
SARIMA_CI_ALPHA = 0.05   # Native SARIMA prediction interval alpha  (SARIMA only)
CHRONOS_CI_ALPHA = 0.05  # Chronos quantile CI alpha                 (Transformer only)


# =============================================================================
# VALIDATION THRESHOLDS
# =============================================================================

# Minimum data size requirements
MIN_TRAIN_SAMPLES = 10      # Minimum samples needed for training
MIN_TEST_SAMPLES = 4        # Minimum samples needed for testing (2x window size)

# Numerical stability thresholds
ZERO_THRESHOLD = 1e-10      # Threshold for considering values as zero (prevents division by zero)

# Date format for parsing
DATE_FORMAT = '%Y-%m-%d'


# =============================================================================
# TIME SERIES CONFIGURATION
# =============================================================================

# Default lookback window for predictions
# Rationale: 15 days captures ~2 weeks of price patterns
DEFAULT_TIME_STEP_BACKWARD = 15

# Default forecast horizon
# Rationale: Single-step ahead prediction is most reliable
DEFAULT_TIME_STEP_FORWARD = 1

# Recursive prediction configuration
DEFAULT_PRED_DAYS = 15      # Number of days to predict recursively


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Log levels for different components
LOG_LEVELS = {
    'default': 'INFO',
    'models': 'INFO',
    'data': 'INFO',
    'portfolio': 'INFO'
}

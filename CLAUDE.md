# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ForecastAGLT is a cryptocurrency price forecasting and portfolio optimization application built with Streamlit. It uses multiple time series forecasting models (LSTM, SARIMA, GMDH, Chronos Transformer) to predict cryptocurrency prices and optimizes portfolio allocations using Modern Portfolio Theory.

## Running the Application

The project uses `uv` for dependency management (see pyproject.toml).

**Install dependencies:**
```bash
uv sync
```

**Run the portfolio optimization app (main page):**
```bash
streamlit run Portfolio_optimization.py
```

**Run the model optimization page:**
```bash
streamlit run pages/1_Model_optimization.py
```

**Docker:**
```bash
# Build and run
docker build -t forecast-app .
docker run -p 8501:8501 forecast-app

# For Mac (with Apple Silicon)
docker build -f Dockerfile_mac -t forecast-app .
```

## Architecture

### Two Main User Flows

1. **Single-Ticker Model Optimization** (`pages/1_Model_optimization.py`)
   - User selects a single cryptocurrency ticker
   - Configures data scaling parameters (scaling strategy, time windows)
   - Runs multiple forecasting models (LSTM, SARIMA, GMDH, Chronos)
   - Compares model performance with metrics (MAPE, RMSE, etc.)
   - Displays predictions vs actual prices

2. **Portfolio Optimization** (`Portfolio_optimization.py`)
   - User configures portfolio parameters via `sidebar_portfolio.py`
   - Fetches top N cryptocurrencies from CoinGecko API
   - Runs forecasting models for each valid ticker
   - Selects non-correlated assets (correlation threshold: 0.9)
   - Optimizes portfolio weights using predicted returns and covariance matrix
   - Validates portfolio performance on validation and test sets

### Core Components

**experiment_runner_for_best_models.py**
- `experiment()` function: Main entry point for single-ticker forecasting
- Downloads data from yfinance, applies scaling strategies, trains multiple models
- Returns plot data, metrics, and trained models
- Supports data scaling strategies: 'average', 'median', 'undersampling' over time windows

**experiment_runner_for_portfolio.py**
- `DataLoader` class:
  - Fetches top N cryptocurrencies from CoinGecko
  - Validates tickers with yfinance
  - Runs experiments for each ticker using `experiment()`
  - Handles date alignment across multiple tickers
  - Filters assets by correlation threshold
  - Splits data into train/validation/test sets

- `Portfolio` class:
  - Implements Modern Portfolio Theory optimization (scipy.optimize)
  - Supports constraints: long-only or allow short positions
  - Can optimize for minimum volatility or target return
  - Calculates portfolio metrics: return, volatility, Sharpe ratio
  - Processes each time period T → T+1 for rolling evaluation

**pages/utils/utils.py**
- `create_dataset()`: Creates time-series sequences for model training
- `make_prediction()`: Unified prediction interface for all model types
- `make_prediction_recursive()`: Recursive multi-step forecasting for LSTM

**sidebar_portfolio.py**
- Configures portfolio parameters: number of assets, investment horizon, scaling strategy, target return, short positions

### Models

The application supports four forecasting model types:
- **LSTM**: TensorFlow/Keras LSTM neural network
- **SARIMA**: Auto ARIMA from pmdarima (auto_arima)
- **GMDH**: Group Method of Data Handling (polynomial networks)
- **Chronos**: Amazon's Chronos-T5-Tiny transformer model (loaded via ChronosPipeline)

All models are trained on the same train/test split and evaluated using MAPE and other metrics.

### Data Flow

1. Download crypto price data from Yahoo Finance via yfinance
2. Optional: Apply time-based scaling (aggregation over D/W/M/Y periods)
3. Create sequences using sliding window (time_step_backward lookback period)
4. Train models and generate predictions
5. For portfolio: align predictions across tickers, compute covariance matrix
6. Optimize portfolio weights using predicted returns and covariance
7. Evaluate portfolio performance on validation and test sets

## Key Configuration Parameters

- `top_n`: Number of top cryptocurrencies to consider
- `num_scale_steps`: Time aggregation window size (e.g., 7 for weekly)
- `scaling_strategy`: 'average', 'median', or 'undersampling'
- `time_step_backward`: Number of historical days used as features (default: 15)
- `correlation_threshold`: Max correlation for asset selection (default: 0.9)
- `target_return`: Optional target return for portfolio optimization
- `allow_short`: Whether to allow short positions in portfolio

## Important Implementation Details

- The Chronos model is cached using `@st.cache_data` to avoid reloading
- `YF_DISABLE_CURL_CFFI` environment variable is set to fix yfinance compatibility
- Portfolio optimization uses SLSQP method from scipy
- Global date alignment ensures all tickers have overlapping train/test periods
- Covariance matrix is computed on training data only
- Portfolio evaluation uses rolling window: optimize at T using predicted T+1, evaluate with actual T+1

## Python Version

Python 3.11+ (see .python-version and pyproject.toml)

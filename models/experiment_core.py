"""
Core experiment functions for cryptocurrency price prediction.

This module contains shared logic for data loading, preprocessing, model training,
prediction generation, and metrics calculation used across multiple experiment runners.
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, Tuple, Optional, Any

# Sklearn imports
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# TensorFlow/Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping

# External libraries
import yfinance as yf
import pmdarima as pm
from gmdh import CriterionType, Criterion, Multi, Combi, Mia, Ria, PolynomialType
from chronos import ChronosPipeline
import torch

# Local imports
from pages.utils.utils import create_dataset, make_prediction
from logging_config import get_logger
import config

logger = get_logger(__name__)


def load_crypto_data(ticker: str, interval: str = config.DEFAULT_INTERVAL) -> pd.DataFrame:
    """
    Load cryptocurrency data from yfinance or fallback to local CSV file.

    Args:
        ticker: Cryptocurrency ticker symbol (e.g., 'BTC', 'ETH')
        interval: Trading interval (e.g., '1d', '1wk', '1mo')

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume

    Raises:
        FileNotFoundError: If data cannot be loaded from yfinance or local file
    """
    int_to_periods = {
        '1m': '5d', '2m': '1mo', '5m': '1mo', '15m': '1mo', '30m': '1mo',
        '60m': '1mo', '90m': '1mo', '1h': '1y', '1d': '10y', '5d': '10y',
        '1wk': '10y', '1mo': '10y', '3mo': '10y'
    }

    import os

    try:
        logger.info(f"Attempting to download data for {ticker}-USD from yfinance...")
        maindf = yf.download(
            tickers=f"{ticker}-USD",
            period='max',
            interval=interval,
            prepost=False,
            repair=True,
            progress=False
        )

        if len(maindf) == 0:
            raise ValueError(f"No data downloaded for {ticker}-USD with interval {interval}")

        logger.info(f"Successfully downloaded {len(maindf)} rows for {ticker}-USD")

    except (FileNotFoundError, ValueError, ConnectionError) as e:
        # Try fallback to local CSV file
        csv_path = f'{ticker}.csv'
        logger.warning(f"Failed to download from yfinance: {e}. Attempting local file: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Could not download data for {ticker}-USD and local file {csv_path} not found. "
                f"Error from yfinance: {e}"
            )

        try:
            maindf = pd.read_csv(csv_path)
            logger.info(f"Successfully loaded {len(maindf)} rows from {csv_path}")
        except Exception as csv_error:
            raise FileNotFoundError(
                f"Failed to read local CSV {csv_path}: {csv_error}"
            )

    # Validate data structure
    if not isinstance(maindf, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(maindf)}")

    if maindf.empty:
        raise ValueError(f"Loaded data for {ticker} is empty")

    # Reset index and ensure Date column exists
    maindf = maindf.reset_index()

    # Check if 'Date' column exists (might be 'date', 'Datetime', etc.)
    date_columns = [col for col in maindf.columns if 'date' in col.lower()]
    if not date_columns:
        raise ValueError(
            f"No date column found in data for {ticker}. "
            f"Available columns: {list(maindf.columns)}"
        )

    # Rename to standard 'Date' if needed
    if 'Date' not in maindf.columns and date_columns:
        maindf = maindf.rename(columns={date_columns[0]: 'Date'})
        logger.info(f"Renamed column '{date_columns[0]}' to 'Date'")

    # Ensure Date column is datetime
    try:
        maindf['Date'] = pd.to_datetime(maindf['Date'])
    except Exception as e:
        raise ValueError(f"Failed to convert Date column to datetime: {e}")

    # Validate required columns exist
    required_columns = ['Date', 'Close']
    missing_columns = [col for col in required_columns if col not in maindf.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns {missing_columns} for {ticker}. "
            f"Available: {list(maindf.columns)}"
        )

    return maindf


def preprocess_data(
    data: pd.DataFrame,
    num_scale_steps: int = 1,
    scaling_strategy: str = 'average',
    scale_step_type: str = 'D'
) -> pd.DataFrame:
    """
    Preprocess and optionally scale time series data.

    Args:
        data: DataFrame with Date and Close columns
        num_scale_steps: Number of steps for scaling (1 = no scaling)
        scaling_strategy: Strategy for aggregation ('average', 'median', 'undersampling')
        scale_step_type: Type of time step ('D', 'W', 'M', 'Y')

    Returns:
        Preprocessed DataFrame with Date and Close columns
    """
    y_overall = data[['Date', 'Close']].copy()

    if num_scale_steps > 1:
        scaling_step_combined = str(num_scale_steps) + scale_step_type
        today = pd.Timestamp.now().normalize()

        if scaling_strategy == 'average':
            y_overall['Interval_End'] = today - (
                (today - y_overall['Date']) // pd.Timedelta(scaling_step_combined)
            ) * pd.Timedelta(scaling_step_combined)
            y_overall = y_overall.groupby('Interval_End')['Close'].mean().reset_index()
            y_overall = y_overall.sort_values('Interval_End')
            y_overall = y_overall.rename({'Interval_End': 'Date'}, axis=1)

        elif scaling_strategy == 'median':
            y_overall['Interval_End'] = today - (
                (today - y_overall['Date']) // pd.Timedelta(scaling_step_combined)
            ) * pd.Timedelta(scaling_step_combined)
            y_overall = y_overall.groupby('Interval_End')['Close'].median().reset_index()
            y_overall = y_overall.sort_values('Interval_End')
            y_overall = y_overall.rename({'Interval_End': 'Date'}, axis=1)

        else:  # undersampling
            y_overall = y_overall.resample(
                on='Date',
                rule=scaling_step_combined,
                origin='end'
            ).last()
            y_overall = y_overall.reset_index()

    # Handle MultiIndex columns if present
    if isinstance(y_overall.columns, pd.MultiIndex):
        y_overall.columns = y_overall.columns.droplevel(1)

    return y_overall


def prepare_train_test_data(
    closedf: pd.DataFrame,
    time_step_backward: int = config.DEFAULT_TIME_STEP_BACKWARD,
    time_step_forward: int = config.DEFAULT_TIME_STEP_FORWARD,
    train_ratio: float = config.TRAIN_RATIO,
    max_samples: int = config.MAX_SAMPLES
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, pd.DataFrame, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Prepare training and testing data with proper scaling and windowing.

    Args:
        closedf: DataFrame with Date and Close columns
        time_step_backward: Number of historical steps for features
        time_step_forward: Number of future steps for target
        train_ratio: Ratio of data to use for training
        max_samples: Maximum number of samples to use

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, scaler, close_stock, train_date_range)

    Raises:
        ValueError: If data size is insufficient for the given parameters
    """
    # Limit data size
    closedf = closedf[-max_samples:]
    close_stock = closedf.copy()

    # Separate dates and data
    dates = closedf['Date'].copy()
    data = closedf[['Close']].copy()

    # Create scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Split into train and test
    training_size = int(len(data) * train_ratio)
    test_size = len(data) - training_size
    min_required_size = 2 * (time_step_backward + time_step_forward)

    # Validate sizes with actionable guidance
    if training_size < time_step_backward + time_step_forward:
        raise ValueError(
            f"Training data has {training_size} samples but {time_step_backward + time_step_forward} required "
            f"({time_step_backward} lookback + {time_step_forward} forecast). "
            f"Solutions: (1) Reduce time_step_backward (currently {time_step_backward}) or time_step_forward (currently {time_step_forward}), "
            f"(2) Increase max_samples (currently {max_samples}), or (3) Get more historical data. "
            f"Total available: {len(data)} samples."
        )
    if test_size <= min_required_size:
        raise ValueError(
            f"Test data has {test_size} samples but {min_required_size} required "
            f"(2x window size for proper evaluation). "
            f"Solutions: (1) Reduce time_step_backward (currently {time_step_backward}) or time_step_forward (currently {time_step_forward}), "
            f"(2) Increase train_ratio to leave more test data (currently {train_ratio:.0%} for training), "
            f"(3) Increase max_samples (currently {max_samples}), or (4) Get more historical data. "
            f"Total available: {len(data)} samples."
        )

    train_data = data.iloc[0:training_size]
    test_data = data.iloc[training_size:]

    # Get train date range
    train_start_date = dates.iloc[0]
    train_end_date = dates.iloc[training_size - 1]

    # Scale data
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    # Create datasets
    X_train, y_train = create_dataset(train_data_scaled, time_step_backward, time_step_forward)
    X_test, y_test = create_dataset(test_data_scaled, time_step_backward, time_step_forward)

    return X_train, y_train, X_test, y_test, scaler, close_stock, (train_start_date, train_end_date)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = config.RANDOM_SEED,
    lstm_units: int = config.LSTM_CONFIG['units'],
    epochs: int = config.LSTM_CONFIG['epochs'],
    batch_size: int = config.LSTM_CONFIG['batch_size'],
    patience: int = config.LSTM_CONFIG['patience'],
    verbose: bool = config.LSTM_CONFIG['verbose']
) -> Tuple[Sequential, Any]:
    """
    Train an LSTM model for time series prediction.

    Args:
        X_train: Training features (samples, time_steps, features)
        y_train: Training targets
        X_test: Test features for validation
        y_test: Test targets for validation
        seed: Random seed for reproducibility
        lstm_units: Number of LSTM units
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        patience: Early stopping patience
        verbose: Whether to show training progress

    Returns:
        Tuple of (trained_model, training_history)
    """
    # Validate input data
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Training data is empty. Cannot train LSTM model.")

    if X_test.size == 0 or y_test.size == 0:
        raise ValueError("Test data is empty. Cannot train LSTM model.")

    if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
        raise ValueError("Training data contains NaN values. Please clean data before training.")

    if np.any(np.isinf(X_train)) or np.any(np.isinf(y_train)):
        raise ValueError("Training data contains infinite values. Please clean data before training.")

    # Reshape if needed (ensure 3D input for LSTM)
    if len(X_train.shape) == 2:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    try:
        model = Sequential()
        model.add(LSTM(
            lstm_units,
            input_shape=(None, 1),
            activation="relu",
            kernel_initializer=initializers.GlorotNormal(seed=seed),
            bias_initializer=initializers.GlorotNormal(seed=seed)
        ))
        model.add(Dense(
            1,
            kernel_initializer=initializers.GlorotNormal(seed=seed),
            bias_initializer=initializers.GlorotNormal(seed=seed)
        ))
        model.compile(loss="mean_squared_error", optimizer="adam")

        callback = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)

        logger.info(f"Training LSTM with {X_train.shape[0]} samples...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=10 if verbose else 0,
            callbacks=[callback]
        )

        # Check if training converged
        final_loss = history.history['loss'][-1]
        if np.isnan(final_loss) or np.isinf(final_loss):
            raise ValueError(f"Training failed: final loss is {final_loss}")

        logger.info(f"LSTM training completed. Final loss: {final_loss:.6f}")

    except Exception as e:
        logger.error(f"Error during LSTM training: {e}")
        raise RuntimeError(f"Failed to train LSTM model: {e}") from e

    return model, history


def train_sarima(
    train_data: np.ndarray,
    time_step_backward: int = config.DEFAULT_TIME_STEP_BACKWARD,
    seasonal: bool = config.SARIMA_CONFIG['seasonal'],
    m: int = config.SARIMA_CONFIG['m'],
    trace: bool = config.SARIMA_CONFIG['trace']
) -> Any:
    """
    Train a SARIMA model using auto_arima for automatic parameter selection.

    Args:
        train_data: Training data (scaled)
        time_step_backward: Maximum lag to consider for p and q parameters
        seasonal: Whether to use seasonal ARIMA
        m: Frequency of series (seasonality period)
        trace: Whether to print optimization trace

    Returns:
        Fitted SARIMA model
    """
    # Validate input data
    if train_data.size == 0:
        raise ValueError("Training data is empty. Cannot train SARIMA model.")

    if np.any(np.isnan(train_data)):
        raise ValueError("Training data contains NaN values. Please clean data before training.")

    if np.any(np.isinf(train_data)):
        raise ValueError("Training data contains infinite values. Please clean data before training.")

    if len(train_data) < 2 * m:
        logger.warning(
            f"Training data size ({len(train_data)}) is less than 2x seasonal period ({2*m}). "
            f"SARIMA may not fit well."
        )

    try:
        logger.info(f"Training SARIMA model with {len(train_data)} samples...")
        arima_model = pm.auto_arima(
            train_data,
            m=m,
            seasonal=seasonal,
            d=None,  # let model determine 'd'
            test='adf',  # use adftest to find optimal 'd'
            start_p=0, start_q=0,
            max_p=min(time_step_backward, len(train_data) // 2 - 1),  # Ensure max_p doesn't exceed data size
            max_q=min(time_step_backward, len(train_data) // 2 - 1),
            D=None,  # let model determine 'D'
            trace=trace,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

        logger.info(f"SARIMA model fitted: {arima_model.order}, seasonal: {arima_model.seasonal_order}")

    except Exception as e:
        logger.error(f"Error during SARIMA training: {e}")
        raise RuntimeError(f"Failed to train SARIMA model: {e}") from e

    return arima_model


def train_gmdh_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    gmdh_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train GMDH models with specified parameters.

    Args:
        X_train: Training features (2D array for GMDH)
        y_train: Training targets
        gmdh_params: Dictionary containing GMDH configuration with keys:
            - algorithms: Dict of algorithm configs, each with:
                - type: One of ['Combi', 'Multi', 'Mia', 'Ria']
                - criterion: CriterionType enum value
                - p_average: int
                - limit: float
                - k_best: int (for Multi, Mia, Ria)
                - polynomial_type: PolynomialType enum (for Mia, Ria)

    Returns:
        Dictionary of trained GMDH models keyed by model name
    """
    # Validate input data
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Training data is empty. Cannot train GMDH models.")

    if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
        raise ValueError("Training data contains NaN values. Please clean data before training.")

    if 'algorithms' not in gmdh_params or not gmdh_params['algorithms']:
        raise ValueError("No GMDH algorithms specified in gmdh_params")

    GMDHs = {'Combi': Combi(), 'Multi': Multi(), 'Mia': Mia(), 'Ria': Ria()}
    models = {}

    for model_name, params in gmdh_params.get('algorithms', {}).items():
        try:
            algo_type = params.get('type')
            if not algo_type or algo_type not in GMDHs:
                logger.error(f"Invalid GMDH algorithm type '{algo_type}' for {model_name}")
                continue

            logger.info(f"Training {model_name} ({algo_type}) with {X_train.shape[0]} samples...")
            model_gmdh = GMDHs[algo_type]

            fit_kwargs = {
                'p_average': params.get('p_average', 1),
                'limit': params.get('limit', 0.0),
                'test_size': 0.3,
                'criterion': Criterion(criterion_type=params['criterion'])
            }

            if algo_type in ['Multi', 'Mia', 'Ria']:
                fit_kwargs['k_best'] = params.get('k_best', 1)

            if algo_type in ['Mia', 'Ria']:
                fit_kwargs['polynomial_type'] = params.get('polynomial_type')

            model_gmdh.fit(X_train, y_train, **fit_kwargs)
            models[model_name] = model_gmdh
            logger.info(f"{model_name} training completed successfully")

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            # Continue with other models even if one fails
            continue

    if not models:
        raise RuntimeError("Failed to train any GMDH models")

    return models


def calculate_all_metrics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    scaler: MinMaxScaler
) -> pd.DataFrame:
    """
    Calculate comprehensive metrics for all models.

    Args:
        y_train: Original training targets (scaled)
        y_test: Original test targets (scaled)
        predictions: Dict mapping model names to (train_pred, test_pred) tuples
        scaler: Scaler used for inverse transformation

    Returns:
        DataFrame with metrics for each model
    """
    # Inverse transform actual values
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    metrics_dict = {}
    metric_names = [
        "Train data RMSE", "Train data MSE", "Train data MAE", "Train data MAPE",
        "Test data RMSE", "Test data MSE", "Test data MAE", "Test data MAPE",
        "Train data R2 score", "Test data R2 score"
    ]

    for model_name, (train_pred, test_pred) in predictions.items():
        metrics = []

        # Train metrics
        metrics.append(math.sqrt(mean_squared_error(original_ytrain, train_pred)))
        metrics.append(mean_squared_error(original_ytrain, train_pred))
        metrics.append(mean_absolute_error(original_ytrain, train_pred))
        metrics.append(mean_absolute_percentage_error(original_ytrain, train_pred))

        # Test metrics
        metrics.append(math.sqrt(mean_squared_error(original_ytest, test_pred)))
        metrics.append(mean_squared_error(original_ytest, test_pred))
        metrics.append(mean_absolute_error(original_ytest, test_pred))
        metrics.append(mean_absolute_percentage_error(original_ytest, test_pred))

        # R2 scores
        metrics.append(r2_score(original_ytrain, train_pred))
        metrics.append(r2_score(original_ytest, test_pred))

        metrics_dict[model_name] = metrics

    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='columns')
    metrics_df.index = metric_names

    return metrics_df


def create_plot_dataframe(
    close_stock: pd.DataFrame,
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    closedf_shape: Tuple[int, int],
    time_step_backward: int,
    time_step_forward: int = 1
) -> pd.DataFrame:
    """
    Create a DataFrame for plotting predictions vs actual values.

    Args:
        close_stock: Original close stock DataFrame with Date and Close columns
        predictions: Dict mapping model names to (train_pred, test_pred) tuples
        closedf_shape: Shape of the closedf used for creating empty arrays
        time_step_backward: Number of backward steps (for lag calculation)
        time_step_forward: Number of forward steps (for lag calculation)

    Returns:
        DataFrame with date, original close, and all model predictions
    """
    lag = time_step_backward + (time_step_forward - 1)

    # Start with date and original close
    plot_data = {
        'date': close_stock['Date'],
        'original_close': close_stock['Close']
    }

    # Add predictions for each model
    for model_name, (train_pred, test_pred) in predictions.items():
        # Create empty arrays for plotting
        trainPredictPlot = np.empty(closedf_shape)
        trainPredictPlot[:, :] = np.nan

        # Validate train predictions shape matches expected range
        train_start_idx = lag
        train_end_idx = len(train_pred) + lag
        if train_end_idx > closedf_shape[0]:
            logger.warning(
                f"Train predictions for {model_name} exceed data size. "
                f"Expected max {closedf_shape[0]}, got {train_end_idx}. Truncating."
            )
            train_end_idx = closedf_shape[0]
            train_pred = train_pred[:train_end_idx - train_start_idx]

        if train_pred.shape[0] > 0:
            trainPredictPlot[train_start_idx:train_end_idx, :] = train_pred

        testPredictPlot = np.empty(closedf_shape)
        testPredictPlot[:, :] = np.nan

        # Validate test predictions shape matches expected range
        test_start_idx = len(train_pred) + (lag * 2)
        test_end_idx = closedf_shape[0]
        expected_test_len = test_end_idx - test_start_idx

        if test_pred.shape[0] != expected_test_len:
            logger.warning(
                f"Test predictions for {model_name} shape mismatch. "
                f"Expected {expected_test_len}, got {test_pred.shape[0]}. "
                f"Adjusting..."
            )
            if test_pred.shape[0] > expected_test_len:
                # Truncate if too long
                test_pred = test_pred[:expected_test_len]
            else:
                # Pad with NaN if too short
                padding = np.full((expected_test_len - test_pred.shape[0], 1), np.nan)
                test_pred = np.vstack([test_pred, padding])

        if test_start_idx < closedf_shape[0] and test_pred.shape[0] > 0:
            testPredictPlot[test_start_idx:test_end_idx, :] = test_pred

        # Convert model name to column-friendly format
        model_suffix = model_name.lower().replace(' ', '_')
        plot_data[f'train_predicted_close_{model_suffix}'] = trainPredictPlot.reshape(1, -1)[0].tolist()
        plot_data[f'test_predicted_close_{model_suffix}'] = testPredictPlot.reshape(1, -1)[0].tolist()

    return pd.DataFrame(plot_data)


def get_chronos_pipeline(cache: bool = False) -> ChronosPipeline:
    """
    Load the Chronos transformer pipeline.

    Args:
        cache: Whether to cache the pipeline (for Streamlit apps)

    Returns:
        Loaded ChronosPipeline
    """
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",  # use "cpu" for CPU inference and "mps" for Apple Silicon
        torch_dtype=torch.bfloat16
    )
    return pipeline


def make_all_predictions(
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_train_gmdh: np.ndarray,
    X_test_gmdh: np.ndarray,
    models: Dict[str, Any],
    scaler: MinMaxScaler,
    time_step_forward: int = 1,
    include_transformer: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate predictions for all trained models.

    Args:
        X_train: LSTM training features (3D)
        X_test: LSTM test features (3D)
        X_train_gmdh: GMDH training features (2D)
        X_test_gmdh: GMDH test features (2D)
        models: Dictionary of trained models
        scaler: Scaler for inverse transformation
        time_step_forward: Number of steps forward to predict
        include_transformer: Whether to include transformer predictions

    Returns:
        Dictionary mapping model names to (train_pred, test_pred) tuples
    """
    predictions = {}

    # LSTM predictions
    if 'LSTM' in models:
        train_pred, test_pred = make_prediction(
            X_train, X_test,
            method='LSTM',
            model=models['LSTM'],
            scaler=scaler,
            time_step_forward=time_step_forward
        )
        predictions['LSTM'] = (train_pred, test_pred)

    # SARIMA predictions
    if 'SARIMA' in models:
        train_pred, test_pred = make_prediction(
            X_train_gmdh, X_test_gmdh,
            method='SARIMA',
            model=models['SARIMA'],
            scaler=scaler,
            time_step_forward=time_step_forward
        )
        predictions['SARIMA'] = (train_pred, test_pred)

    # GMDH predictions
    gmdh_models = {k: v for k, v in models.items() if k.startswith('GMDH')}
    for model_name, model in gmdh_models.items():
        train_pred, test_pred = make_prediction(
            X_train_gmdh, X_test_gmdh,
            method='GMDH',
            model=model,
            scaler=scaler,
            time_step_forward=time_step_forward
        )
        predictions[model_name] = (train_pred, test_pred)

    # Transformer predictions
    if include_transformer and 'Transformer' in models:
        train_pred, test_pred = make_prediction(
            X_train_gmdh, X_test_gmdh,
            method='Transformer',
            model=models['Transformer'],
            scaler=scaler,
            time_step_forward=time_step_forward
        )
        predictions['Transformer'] = (train_pred, test_pred)

    return predictions

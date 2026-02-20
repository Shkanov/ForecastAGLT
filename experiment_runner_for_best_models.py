"""
Experiment runner for best models.

This module provides the main experiment function for running cryptocurrency
price prediction experiments with multiple models (LSTM, SARIMA, GMDH, Transformer).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from logging_config import get_logger
from gmdh import CriterionType, PolynomialType
from models.experiment_core import (
    load_crypto_data,
    preprocess_data,
    prepare_train_test_data,
    train_lstm,
    train_sarima,
    train_gmdh_models,
    get_chronos_pipeline,
    make_all_predictions,
    calculate_all_metrics,
    create_plot_dataframe
)


def experiment(ticker, num_scale_steps, scaling_strategy, time_step_backward):
    """
    Run a complete cryptocurrency price prediction experiment.

    Args:
        ticker: Cryptocurrency ticker symbol (e.g., 'BTC', 'ETH')
        num_scale_steps: Number of steps for data scaling (1 = no scaling)
        scaling_strategy: Strategy for aggregation ('average', 'median', 'undersampling')
        time_step_backward: Number of historical steps for features

    Returns:
        Tuple of (plotdf, metrics_df, models_dict)
            - plotdf: DataFrame with predictions for plotting
            - metrics_df: DataFrame with model performance metrics
            - models_dict: Dictionary of trained models
    """
    logger = get_logger(__name__)

    pd.options.display.float_format = '{:20,.4f}'.format
    seed = 42
    interval = '1d'
    time_step_forward = 1
    pred_days = 15
    recursive_pred = False
    transformer = True
    GMDH = True

    # Load and preprocess data
    logger.info(f"Loading data for {ticker}")
    maindf = load_crypto_data(ticker, interval)
    logger.info(f'Total number of days present in the dataset: {maindf.shape[0]}')
    logger.info(f'Total number of fields present in the dataset: {maindf.shape[1]}')
    logger.debug(f'Dataset head:\n{maindf.head()}')

    # Preprocess and scale data
    y_overall = preprocess_data(maindf, num_scale_steps, scaling_strategy, scale_step_type='D')


    # Initial plot
    fig, ax = plt.subplots()
    ax.plot(y_overall['Close'], label='Stock Close Price')
    ax.legend()
    ax.set_title(f'Динамика цены закрытия для {ticker}')
    plt.close(fig)  # Close figure to prevent memory leak

    # Prepare train/test data
    closedf = y_overall[['Date', 'Close']].dropna()
    logger.info(f"Shape of close dataframe: {closedf.shape}")

    X_train, y_train, X_test, y_test, scaler, close_stock, train_date_range = prepare_train_test_data(
        closedf,
        time_step_backward=time_step_backward,
        time_step_forward=time_step_forward,
        train_ratio=0.70,
        max_samples=1000
    )

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

    # Keep GMDH copy before reshaping for LSTM
    X_train_gmdh = X_train.copy()
    X_test_gmdh = X_test.copy()

    # Train LSTM model
    logger.info("Training LSTM model...")
    lstm_model, history = train_lstm(
        X_train, y_train, X_test, y_test,
        seed=seed,
        lstm_units=10,
        epochs=100,
        batch_size=32,
        patience=30,
        verbose=False
    )

    # Train SARIMA model
    logger.info("Training SARIMA model...")
    # Get scaled train data for SARIMA
    train_data_scaled = scaler.transform(close_stock[['Close']].iloc[:int(len(close_stock) * 0.70)])
    arima_model = train_sarima(
        train_data_scaled,
        time_step_backward=time_step_backward,
        seasonal=True,
        m=12,
        trace=True
    )
    logger.info(f"ARIMA model summary:\n{arima_model.summary()}")

    # Train GMDH models
    models_dict = {'LSTM': lstm_model, 'SARIMA': arima_model}

    if GMDH:
        logger.info("Training GMDH models...")
        gmdh_params = {
            'algorithms': {
                'GMDH_1': {
                    'type': 'Multi',
                    'criterion': CriterionType.REGULARITY,
                    'p_average': 1,
                    'limit': 0.,
                    'k_best': 1,
                    'polynomial_type': PolynomialType.LINEAR
                },
                'GMDH_2': {
                    'type': 'Ria',
                    'criterion': CriterionType.REGULARITY,
                    'p_average': 1,
                    'limit': 0.,
                    'k_best': 3,
                    'polynomial_type': PolynomialType.QUADRATIC
                }
            }
        }
        gmdh_models = train_gmdh_models(X_train_gmdh, y_train, gmdh_params)
        for name, model in gmdh_models.items():
            logger.info(f"{name}: {model.get_best_polynomial()}")
        models_dict.update(gmdh_models)

    # Load Transformer pipeline if needed
    if transformer:
        logger.info("Loading Transformer pipeline...")
        pipeline = get_chronos_pipeline()
        models_dict['Transformer'] = pipeline

    # Plot training history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    fig, ax = plt.subplots()
    ax.plot(epochs, loss, 'r', label='Training loss')
    ax.plot(epochs, val_loss, 'b', label='Validation loss')
    ax.legend()
    ax.set_title('Потери на обучении и валидации')
    ax.plot()
    plt.close(fig)  # Close figure to prevent memory leak

    # Make predictions for all models
    logger.info("Generating predictions...")
    predictions = make_all_predictions(
        X_train=X_train.reshape(X_train.shape[0], X_train.shape[1], 1),
        X_test=X_test.reshape(X_test.shape[0], X_test.shape[1], 1),
        X_train_gmdh=X_train_gmdh,
        X_test_gmdh=X_test_gmdh,
        models=models_dict,
        scaler=scaler,
        time_step_forward=time_step_forward,
        include_transformer=transformer
    )

    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics_df = calculate_all_metrics(y_train, y_test, predictions, scaler)
    logger.info(f"Metrics DataFrame:\n{metrics_df}")
    # Create plot dataframe
    logger.info("Creating plot dataframe...")
    closedf_for_plot = close_stock[['Close']].iloc[-1000:]  # Match the max_samples used
    plotdf = create_plot_dataframe(
        close_stock=close_stock,
        predictions=predictions,
        closedf_shape=closedf_for_plot.shape,
        time_step_backward=time_step_backward,
        time_step_forward=time_step_forward
    )

    # Create comparison plot
    fig, ax = plt.subplots()
    ax.plot(plotdf['date'], plotdf['original_close'], label='Оригинальная цена закрытия')

    # Plot predictions for each model
    for model_name in predictions.keys():
        model_suffix = model_name.lower().replace(' ', '_')
        if f'train_predicted_close_{model_suffix}' in plotdf.columns:
            ax.plot(plotdf['date'], plotdf[f'train_predicted_close_{model_suffix}'],
                    label=f'Предсказанная цена закрытия на тренировке {model_name}')
            ax.plot(plotdf['date'], plotdf[f'test_predicted_close_{model_suffix}'],
                    label=f'Предсказанная цена закрытия на тесте {model_name}')

    ax.legend()
    ax.set_title("Сравнение исходных и смоделированных цен")
    plt.close(fig)  # Close figure to prevent memory leak

    return plotdf, metrics_df, models_dict
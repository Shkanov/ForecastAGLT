import numpy as np
from typing import Literal, Tuple, List, Optional
import torch

def create_dataset(dataset: np.ndarray, time_step_backward: int = 1,
                   time_step_forward: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window dataset for time series prediction.

    Args:
        dataset: 2D numpy array of shape (n_samples, n_features) containing time series data
        time_step_backward: Number of historical time steps to use as features (lookback window)
        time_step_forward: Number of future time steps to predict (forecast horizon)

    Returns:
        Tuple of (X, y) where:
            - X: Array of shape (n_windows, time_step_backward) containing feature windows
            - y: Array of shape (n_windows,) containing target values

    Example:
        If dataset = [[1], [2], [3], [4], [5]] with time_step_backward=2, time_step_forward=1:
        X = [[1, 2], [2, 3], [3, 4]]
        y = [3, 4, 5]
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step_backward - (time_step_forward - 1)):
        a = dataset[i:(i + time_step_backward), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step_backward + (time_step_forward - 1), 0])
    return np.array(dataX), np.array(dataY)

def make_prediction(X_train: np.ndarray, X_test: np.ndarray,
                    method: Literal['LSTM', 'GMDH', 'Transformer', 'SARIMA'],
                    model, scaler, time_step_forward: int) -> Tuple[np.ndarray, np.ndarray]:
    if method == 'LSTM':
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        return train_predict, test_predict
    elif method == 'SARIMA':
        # SARIMA models handle their own history internally and don't use windowed features
        # We make in-sample predictions for training data and out-of-sample for test data
        # NOTE: X_train and X_test represent sliding windows, but SARIMA doesn't use them
        # Instead, we predict directly from the model's internal state

        train_predict_arima = []
        test_predict_arima = []

        # For training set: Use in-sample predictions
        # Get predictions for the same length as training windows
        n_train = len(X_train)
        train_preds = model.predict_in_sample()  # In-sample predictions

        # Take the last n_train predictions to match our windowed dataset
        if len(train_preds) >= n_train:
            train_predict_arima = train_preds[-n_train:]
        else:
            # If not enough predictions, pad with the available ones
            train_predict_arima = np.pad(train_preds, (n_train - len(train_preds), 0),
                                         mode='edge')

        # For test set: Make out-of-sample forecasts
        # Predict n_test steps ahead from the end of training data
        n_test = len(X_test)
        test_preds = model.predict(n_periods=n_test)
        test_predict_arima = test_preds

        train_predict_arima = np.array(train_predict_arima)
        test_predict_arima = np.array(test_predict_arima)
        train_predict_arima = scaler.inverse_transform(train_predict_arima.reshape(-1, 1))
        test_predict_arima = scaler.inverse_transform(test_predict_arima.reshape(-1, 1))
        return train_predict_arima, test_predict_arima
    elif method == 'GMDH':
        train_predict_gmdh = model.predict(X_train)
        test_predict_gmdh = model.predict(X_test)
        train_predict_gmdh = scaler.inverse_transform(train_predict_gmdh.reshape(-1, 1))
        test_predict_gmdh = scaler.inverse_transform(test_predict_gmdh.reshape(-1, 1))
        return train_predict_gmdh, test_predict_gmdh
    elif method == 'Transformer':
        X_train_context = torch.tensor(X_train)
        X_test_context = torch.tensor(X_test)
        X_train_forecast = model.predict(
            X_train_context,
            time_step_forward,
            num_samples=3,
            temperature=1.0,
            top_k=50,
            top_p=1.0)
        X_test_forecast = model.predict(
            X_test_context,
            time_step_forward,
            num_samples=3,
            temperature=1.0,
            top_k=50,
            top_p=1.0)
        X_train_forecast_median = np.quantile(X_train_forecast.numpy(), 0.5, axis=1)[:, -1]
        X_test_forecast_median = np.quantile(X_test_forecast.numpy(), 0.5, axis=1)[:, -1]
        X_train_forecast_median = scaler.inverse_transform(X_train_forecast_median.reshape(-1, 1))
        X_test_forecast_median = scaler.inverse_transform(X_test_forecast_median.reshape(-1, 1))
        return X_train_forecast_median, X_test_forecast_median





def make_prediction_recursive(test_data: np.ndarray,
                              method: Literal['LSTM', 'GMDH', 'Transformer', 'SARIMA'],
                              model, scaler, pred_days: int, time_step_backward: int) -> np.ndarray:
    """
    Make recursive predictions for future time steps.

    Recursively predicts future values by using each prediction as input for the next step.

    Args:
        test_data: Scaled test data array
        method: Model type to use for predictions
        model: Trained model instance
        scaler: MinMaxScaler instance for inverse transformation
        pred_days: Number of days to predict into the future
        time_step_backward: Number of historical steps to use as features

    Returns:
        Array of shape (pred_days, 1) containing predictions in original scale
    """
    if method == 'LSTM':
        x_input_lstm = test_data[len(test_data) - time_step_backward:].reshape(1, -1)
        temp_input_lstm = list(x_input_lstm)
        temp_input_lstm = temp_input_lstm[0].tolist()
        lst_output_lstm = []
        n_steps = time_step_backward
        i = 0
        while (i < pred_days):
            if (len(temp_input_lstm) > time_step_backward):

                x_input_lstm = np.array(temp_input_lstm[1:])
                x_input_lstm = x_input_lstm.reshape(1, -1)
                x_input_lstm = x_input_lstm.reshape((1, n_steps, 1))

                yhat_lstm = model.predict(x_input_lstm, verbose=0)
                temp_input_lstm.extend(yhat_lstm[0].tolist())
                temp_input_lstm = temp_input_lstm[1:]
                lst_output_lstm.extend(yhat_lstm.tolist())
                i = i + 1
            else:
                x_input_lstm = x_input_lstm.reshape((1, n_steps, 1))
                yhat_lstm = model.predict(x_input_lstm, verbose=0)
                temp_input_lstm.extend(yhat_lstm[0].tolist())
                lst_output_lstm.extend(yhat_lstm.tolist())
                i = i + 1

        lst_output_lstm = scaler.inverse_transform(lst_output_lstm)
        return lst_output_lstm
    elif method == 'SARIMA':
        # Use the already-fitted model to predict future values
        # Don't refit (fit_predict causes data leakage) - just predict
        lst_output_arima = model.predict(n_periods=pred_days, return_conf_int=False)
        lst_output_arima = scaler.inverse_transform(lst_output_arima.reshape(-1, 1))
        return lst_output_arima
    elif method == 'GMDH':
        x_input_gmdh = test_data[len(test_data) - time_step_backward:].reshape(1, -1)
        temp_input_gmdh = list(x_input_gmdh)
        temp_input_gmdh = temp_input_gmdh[0].tolist()
        lst_output_gmdh = []
        n_steps = time_step_backward
        i = 0
        while (i < pred_days):
            if (len(temp_input_gmdh) > time_step_backward):
                x_input_gmdh = np.array(temp_input_gmdh[1:])
                x_input_gmdh = x_input_gmdh.reshape(1, -1)
                yhat_gmdh = model.predict(x_input_gmdh)
                temp_input_gmdh.extend(yhat_gmdh.tolist())
                temp_input_gmdh = temp_input_gmdh[1:]
                lst_output_gmdh.extend(yhat_gmdh.tolist())
                i = i + 1
            else:
                x_input_gmdh = x_input_gmdh.reshape((1, n_steps, 1))
                yhat_gmdh = model.predict(x_input_gmdh[0].reshape(1, -1))
                temp_input_gmdh.extend(yhat_gmdh.tolist())
                lst_output_gmdh.extend(yhat_gmdh.tolist())
                i = i + 1
        lst_output_gmdh = scaler.inverse_transform(np.array(lst_output_gmdh).reshape(-1, 1))
        return lst_output_gmdh
    elif method == 'Transformer':
        x_input_transformer = test_data[len(test_data) - time_step_backward:].reshape(1, -1)
        x_input_transformer = torch.tensor(x_input_transformer)
        lst_output_forecast = model.predict(
            x_input_transformer,
            pred_days,
            num_samples=3,
            temperature=1.0,
            top_k=50,
            top_p=1.0)
        X_train_forecast_median = np.quantile(lst_output_forecast.numpy(), 0.5, axis=1)  # [:, -1]
        lst_output_transformer = scaler.inverse_transform(X_train_forecast_median.reshape(-1, 1))
        return lst_output_transformer

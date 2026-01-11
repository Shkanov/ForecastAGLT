import numpy as np
from typing import Literal
import torch
from typing import List

def create_dataset(dataset, time_step_backward = 1, time_step_forward = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step_backward - (time_step_forward - 1)):
        a = dataset[i:(i + time_step_backward), 0]  ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step_backward + (time_step_forward - 1), 0])
    return np.array(dataX), np.array(dataY)

def make_prediction(X_train: np.ndarray, X_test: np.ndarray,
                    method: Literal['LSTM', 'GMDH', 'Transformer', 'SARIMA'],
                    model, scaler, time_step_forward: None) -> np.ndarray:
    if method == 'LSTM':
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        return train_predict, test_predict
    elif method == 'SARIMA':
        train_predict_arima = []
        test_predict_arima = []
        for sample in X_train:
            train_predict_arima.append(
                model.fit_predict(sample, n_periods=time_step_forward, return_conf_int=False)[-1])
        train_predict_arima = np.array(train_predict_arima)
        for sample in X_test:
            test_predict_arima.append(
                model.fit_predict(sample, n_periods=time_step_forward, return_conf_int=False)[-1])
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
                              model, scaler, pred_days: None, time_step_backward: None) -> List[int]:
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
        x_input_arima = test_data[len(test_data) - time_step_backward:]
        n_steps = time_step_backward
        lst_output_arima = model.fit_predict(x_input_arima, n_periods=pred_days, return_conf_int=False)  # [-1]
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

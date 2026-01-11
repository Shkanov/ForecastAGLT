import requests
from experiment_runner_for_best_models import experiment
from datetime import datetime
from tqdm import tqdm
import numpy as np
import scipy.optimize as sco


class DataLoader():
    def __init__(self, correlation_threshold: float = 0.9):
        self.correlation_threshold = correlation_threshold
    # Function to get top N cryptocurrency tickers
    def get_top_crypto_tickers(self, n):

        url = 'https://api.coingecko.com/api/v3/coins/markets'
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': n,
            'page': 1,
            'sparkline': 'false'
        }
        response = requests.get(url, params=params)
        data = response.json()
        tickers = [coin['symbol'].upper() for coin in data]
        return tickers


    # Function to validate if a ticker is compatible with yfinance
    def validate_ticker(self, ticker):
        import yfinance as yf
        try:
            ticker += '-USD'
            info = yf.Ticker(ticker).info
            return bool(info)  # Returns True if info is not empty
        except Exception:
            return False

    def experiment_data(self, top_n: int = 3, num_scale_steps: int = 1, scaling_strategy: str = 'average', time_step_backward: int = 15):
        # Retrieve top N tickers
        #top_n = 10
        self.tickers = self.get_top_crypto_tickers(top_n)
        # Validate tickers for compatibility with yfinance
        self.valid_tickers = [ticker for ticker in self.tickers if self.validate_ticker(ticker)]
        print("Compatible tickers for yfinance:", len(self.valid_tickers))
        self.invalid_tickers = []

        # Run experiments for each valid ticker
        self.tickers_dict = {}
        for ticker in self.valid_tickers:
            try:
                self.tickers_dict[ticker] = {}
                plot_df, metrics_df, models_dict = experiment(ticker=ticker, num_scale_steps=num_scale_steps,
                                                              scaling_strategy=scaling_strategy, time_step_backward=time_step_backward)
                self.tickers_dict[ticker]['plot_df'] = plot_df
                self.tickers_dict[ticker]['metrics_df'] = metrics_df
                self.tickers_dict[ticker]['models_dict'] = models_dict
            except AssertionError as e:  # Или другой конкретный тип ошибки
                print('EXCEPTION ', str(e), ticker)
                self.invalid_tickers.append(ticker)
                continue

        for invalid_ticker in self.invalid_tickers:
            self.valid_tickers.remove(invalid_ticker)

        # Mapping for prediction columns
        test_predictions_model_mapper = {
            'SARIMA': 'test_predicted_close_arima',
            'LSTM': 'test_predicted_close',
            'GMDH_1': 'test_predicted_close_gmdh_1',
            'GMDH_2': 'test_predicted_close_gmdh_2',
            'Transformer': 'test_predicted_close_transformer'
        }
        train_predictions_model_mapper = {
            'SARIMA': 'train_predicted_close_arima',
            'LSTM': 'train_predicted_close',
            'GMDH_1': 'train_predicted_close_gmdh_1',
            'GMDH_2': 'train_predicted_close_gmdh_2',
            'Transformer': 'train_predicted_close_transformer'
        }

        # Determine global training and testing periods
        self.global_min_date = datetime(2000, 1, 1, 0, 0)
        self.global_max_date = datetime.now()
        for ticker in self.valid_tickers:
            train_last_valid_index = self.tickers_dict[ticker]['plot_df']['train_predicted_close_arima'].last_valid_index()
            train_last_date = self.tickers_dict[ticker]['plot_df'].loc[train_last_valid_index, 'date']
            if train_last_date < self.global_max_date:
                self.global_max_date = train_last_date

            test_first_valid_index = self.tickers_dict[ticker]['plot_df']['test_predicted_close_arima'].first_valid_index()
            test_first_date = self.tickers_dict[ticker]['plot_df'].loc[test_first_valid_index, 'date']
            if test_first_date > self.global_min_date:
                self.global_min_date = test_first_date

            print(train_last_date, train_last_valid_index, test_first_date, test_first_valid_index)

        print(self.global_min_date , self.global_max_date)

        # Collect predictions for the global periods
        self.train_predictions_df_list = []
        self.test_predictions_df_list = []
        self.actual_prices_train = []
        self.actual_prices_test = []
        for ticker in tqdm(self.valid_tickers):
            best_model = self.tickers_dict[ticker]['metrics_df'].T.sort_values(by='Test data MAPE', ascending=True).index[0]
            train_predictions = self.tickers_dict[ticker]['plot_df'][['date', train_predictions_model_mapper[best_model]]]
            train_predictions = train_predictions[train_predictions['date'] <= self.global_max_date]
            train_predictions.rename(columns={train_predictions_model_mapper[best_model]: ticker}, inplace=True)
            self.train_predictions_df_list.append(train_predictions)

            actual_train = self.tickers_dict[ticker]['plot_df'][['date', 'original_close']]
            actual_train = actual_train[actual_train['date'] <= self.global_max_date]
            actual_train.rename(columns={'original_close': ticker}, inplace=True)
            self.actual_prices_train.append(actual_train)

            test_predictions = self.tickers_dict[ticker]['plot_df'][['date', test_predictions_model_mapper[best_model]]]
            test_predictions = test_predictions[test_predictions['date'] >= self.global_min_date]
            test_predictions.rename(columns={test_predictions_model_mapper[best_model]: ticker}, inplace=True)
            self.test_predictions_df_list.append(test_predictions)

            actual_test = self.tickers_dict[ticker]['plot_df'][['date', 'original_close']]
            actual_test = actual_test[actual_test['date'] >= self.global_min_date]
            actual_test.rename(columns={'original_close': ticker}, inplace=True)
            self.actual_prices_test.append(actual_test)

        self.selected_features = [self.valid_tickers[0]]
        #correlation_threshold = 0.9
        for idx, feature in enumerate(self.valid_tickers):
            if idx == 0:
                continue
            print(idx, feature)
            tmp = self.train_predictions_df_list[0].merge(self.train_predictions_df_list[idx], on='date', how='inner')
            # Вычисляем корреляцию нового признака с уже выбранными
            correlations = [abs(tmp[feature].corr(tmp[sel_feature])) for sel_feature in self.selected_features]
            print(correlations)
            max_correlation = max(correlations)

            # Добавляем признак, если максимальная корреляция не превышает порог
            if max_correlation < self.correlation_threshold:
                self.selected_features.append(feature)
                self.train_predictions_df_list[0] = self.train_predictions_df_list[0].merge(self.train_predictions_df_list[idx], on='date', how='inner')
                self.actual_prices_train[0] = self.actual_prices_train[0].merge(self.actual_prices_train[idx], on='date', how='inner')
                self.test_predictions_df_list[0] = self.test_predictions_df_list[0].merge(self.test_predictions_df_list[idx], on='date', how='inner')
                self.actual_prices_test[0] = self.actual_prices_test[0].merge(self.actual_prices_test[idx], on='date', how='inner')
        print(self.selected_features)

        selected_features_and_date = ['date'] + self.selected_features
        print(selected_features_and_date)

        # Calculate covariance matrix for the training period
        train_data = self.train_predictions_df_list[0].drop(columns=['date']).astype(float)
        self.cov_matrix = train_data[self.selected_features].cov()
        print("Covariance matrix for the training period:")
        print(self.cov_matrix)

        # Split the global test period into validation and test sets
        self.validation_size = int(len(self.test_predictions_df_list[0][selected_features_and_date]) * 0.5)
        self.validation_data = self.test_predictions_df_list[0][selected_features_and_date].iloc[:self.validation_size]
        self.validation_actual = self.actual_prices_test[0][selected_features_and_date].iloc[:self.validation_size]
        self.test_data = self.test_predictions_df_list[0][selected_features_and_date].iloc[self.validation_size:]
        self.test_actual = self.actual_prices_test[0][selected_features_and_date].iloc[self.validation_size:]

        # Проверка положительной определённости
        if np.any(np.linalg.eigvals(self.cov_matrix) <= 0):
            raise ValueError("Ковариационная матрица не является положительно определённой.")

        return self.cov_matrix, self.validation_data, self.validation_actual, self.test_data, self.test_actual, self.train_predictions_df_list, self.actual_prices_train, self.test_predictions_df_list, self.actual_prices_test, self.tickers_dict


class Portfolio():

    def calculate_portfolio_metrics(self, weights, returns, cov_matrix):
        portfolio_return = np.dot(weights, returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_return, portfolio_volatility

    def optimize(self, returns, cov_matrix, target_return=None, allow_short=False):
        num_assets = len(returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        if allow_short:
            bounds = tuple((-1, 1) for _ in range(num_assets))  # Allow short positions
        else:
            bounds = tuple((0, 1) for _ in range(num_assets))  # Long-only portfolio
        initial_weights = num_assets * [1. / num_assets]

        if target_return is not None:
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                           {'type': 'eq', 'fun': lambda x: np.dot(x, returns) - target_return})

        result = sco.minimize(
            lambda w: self.calculate_portfolio_metrics(w, returns, cov_matrix)[1],
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x

    def process_period(self, data, actual_data, cov_matrix, target_return=None, allow_short=False):
        # Forecast and optimize portfolio for each point T -> T+1 in validation and test data
        realized_returns = []
        predicted_returns = []
        realized_volatilities = []
        predicted_volatilities = []
        for i in range(len(data) - 1):
            current_data = data.iloc[i:i + 2]  # Include current day and prediction for next day
            actual_current_data = actual_data.iloc[i:i + 2]  # Actual prices for T and T+1
            # Calculate predicted return using actual price at T and predicted price at T+1
            predicted_return = (current_data.drop(columns=['date']).iloc[1]-
                                actual_current_data.drop(columns=['date']).iloc[0]) / actual_current_data.drop(columns=['date']).iloc[0]
            # Optimize portfolio based on predicted returns
            self.weights = self.optimize(predicted_return, cov_matrix, target_return=target_return,
                                         allow_short=allow_short)
            pred_return, pred_volatility = self.calculate_portfolio_metrics(weights=self.weights, returns=predicted_return,
                                                                       cov_matrix=cov_matrix)
            # Compute realized return using actual prices for T and T+1
            realized_return = (actual_current_data.drop(columns=['date']).iloc[1] -
                               actual_current_data.drop(columns=['date']).iloc[0]) / actual_current_data.drop(columns=['date']).iloc[0]

            real_return, real_volatility = self.calculate_portfolio_metrics(weights=self.weights, returns=realized_return,
                                                                               cov_matrix=cov_matrix)
            realized_returns.append(real_return)
            predicted_returns.append(pred_return)
            realized_volatilities.append(real_volatility)
            predicted_volatilities.append(pred_volatility)
        return predicted_returns, realized_returns, predicted_volatilities, realized_volatilities


    # Calculate accuracy metrics for validation and test sets
    def calculate_accuracy(self, predicted, realized):
        return np.mean(np.abs(np.array(predicted) - np.array(realized))) / np.mean(realized)


    # Calculate Sharpe ratio deviation
    def calculate_sharpe_ratio_deviation(self, predicted_returns, realized_returns, predicted_vol, realized_vol):
        predicted_sharpe = np.mean(predicted_returns) / np.mean(predicted_vol)
        realized_sharpe = np.mean(realized_returns) / np.mean(realized_vol)
        return abs(predicted_sharpe - realized_sharpe)

    def optimize_portfolio(self, cov_matrix, validation_data, validation_actual, test_data, test_actual, target_return: int | None = None, allow_short: bool = False):

        # Calculate validation metrics
        self.val_pred_returns, self.val_realized_returns, self.val_pred_vol, self.val_realized_vol = self.process_period(data=validation_data,
                                                                                                actual_data=validation_actual,
                                                                                                cov_matrix=cov_matrix,
                                                                                                target_return=target_return,
                                                                                                allow_short=allow_short)
        self.test_pred_returns, self.test_realized_returns, self.test_pred_vol, self.test_realized_vol = self.process_period(data=test_data,
                                                                                                    actual_data=test_actual,
                                                                                                    cov_matrix=cov_matrix,
                                                                                                    target_return=target_return,
                                                                                                    allow_short=allow_short)

        #print(self.val_pred_returns, self.val_realized_returns, self.val_pred_vol, self.val_realized_vol)
        #print(self.test_pred_returns, self.test_realized_returns, self.test_pred_vol, self.test_realized_vol)


        self.val_return_accuracy = self.calculate_accuracy(self.val_pred_returns, self.val_realized_returns)
        self.val_volatility_accuracy = self.calculate_accuracy(self.val_pred_vol, self.val_realized_vol)
        self.val_sharpe_deviation = self.calculate_sharpe_ratio_deviation(self.val_pred_returns, self.val_realized_returns, self.val_pred_vol, self.val_realized_vol)
        self.val_sum_pred_returns = np.sum(self.val_pred_returns)
        self.val_sum_realized_returns = np.sum(self.val_realized_returns)

        self.test_return_accuracy = self.calculate_accuracy(self.test_pred_returns, self.test_realized_returns)
        self.test_volatility_accuracy = self.calculate_accuracy(self.test_pred_vol, self.test_realized_vol)
        self.test_sharpe_deviation = self.calculate_sharpe_ratio_deviation(self.test_pred_returns, self.test_realized_returns, self.test_pred_vol, self.test_realized_vol)
        self.test_sum_pred_returns = np.sum(self.test_pred_returns)
        self.test_sum_realized_returns = np.sum(self.test_realized_returns)

        print(f"Validation Return Accuracy: {self.val_return_accuracy}")
        print(f"Validation Volatility Accuracy: {self.val_volatility_accuracy}")
        print(f"Validation Sharpe Ratio Deviation: {self.val_sharpe_deviation}")
        print(f"Validation Pred Return Sum: {self.val_sum_pred_returns}")
        print(f"Validation Actual Return Sum: {self.val_sum_realized_returns}")

        print(f"Test Return Accuracy: {self.test_return_accuracy}")
        print(f"Test Volatility Accuracy: {self.test_volatility_accuracy}")
        print(f"Test Sharpe Ratio Deviation: {self.test_sharpe_deviation}")
        print(f"Test Pred Return Sum: {self.test_sum_pred_returns}")
        print(f"Test Actual Return Sum: {self.test_sum_realized_returns}")

        #return val_return_accuracy, val_volatility_accuracy, val_sharpe_deviation, np.sum(val_pred_vol), np.sum(val_realized_returns), test_return_accuracy, test_volatility_accuracy, test_sharpe_deviation, np.sum(test_pred_vol), np.sum(test_realized_returns)

"""
Portfolio optimization module.

This module builds portfolios from individual asset predictions.
It depends on experiment_runner_for_best_models to generate predictions
for each ticker, then optimizes portfolio weights based on those predictions.

Dependency Direction: portfolio -> best_models -> experiment_core
This is a one-way dependency chain, not circular.
"""

import requests
import config
from experiment_runner_for_best_models import experiment
from datetime import datetime
from tqdm import tqdm
import numpy as np
import scipy.optimize as sco
from scipy.stats import norm
from logging_config import get_logger

logger = get_logger(__name__)


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
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, list) or len(data) == 0:
                logger.error("API returned unexpected data format or empty list")
                raise ValueError("No cryptocurrency data returned from API")

            tickers = [coin['symbol'].upper() for coin in data]
            logger.info(f"Successfully fetched {len(tickers)} tickers from CoinGecko API")
            return tickers

        except requests.exceptions.Timeout:
            logger.error("API request timed out after 10 seconds")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing API response: {e}")
            raise


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
        self.time_step_backward = time_step_backward
        # Retrieve top N tickers
        #top_n = 10
        self.tickers = self.get_top_crypto_tickers(top_n)
        # Validate tickers for compatibility with yfinance
        self.valid_tickers = [ticker for ticker in self.tickers if self.validate_ticker(ticker)]
        logger.info(f"Compatible tickers for yfinance: {len(self.valid_tickers)}")
        self.invalid_tickers = []

        # Run experiments for each valid ticker
        self.tickers_dict = {}
        for ticker in self.valid_tickers:
            try:
                self.tickers_dict[ticker] = {}
                plot_df, metrics_df, models_dict, close_stock, scaler, maindf, model_hyperparams = experiment(
                    ticker=ticker, num_scale_steps=num_scale_steps,
                    scaling_strategy=scaling_strategy, time_step_backward=time_step_backward)
                self.tickers_dict[ticker]['plot_df'] = plot_df
                self.tickers_dict[ticker]['metrics_df'] = metrics_df
                self.tickers_dict[ticker]['models_dict'] = models_dict
                self.tickers_dict[ticker]['close_stock'] = close_stock
                self.tickers_dict[ticker]['scaler'] = scaler
                self.tickers_dict[ticker]['maindf'] = maindf
                self.tickers_dict[ticker]['model_hyperparams'] = model_hyperparams
            except (AssertionError, FileNotFoundError, ValueError, Exception) as e:
                logger.error(f'Skipping {ticker}: {str(e)}')
                self.invalid_tickers.append(ticker)
                continue

        for invalid_ticker in self.invalid_tickers:
            self.valid_tickers.remove(invalid_ticker)

        def get_column_name(model_name: str, prefix: str) -> str:
            """Generate plot_df column name from model name using the same rule as create_plot_dataframe."""
            suffix = model_name.lower().replace(' ', '_')
            return f"{prefix}_{suffix}"

        # Determine global training and testing periods
        self.global_min_date = datetime(2000, 1, 1, 0, 0)
        self.global_max_date = datetime.now()
        for ticker in self.valid_tickers:
            # Use SARIMA predictions (always present) to determine global date boundaries
            plot_df = self.tickers_dict[ticker]['plot_df']
            train_last_valid_index = plot_df['train_predicted_close_sarima'].last_valid_index()
            train_last_date = plot_df.loc[train_last_valid_index, 'date']
            if train_last_date < self.global_max_date:
                self.global_max_date = train_last_date

            test_first_valid_index = plot_df['test_predicted_close_sarima'].first_valid_index()
            test_first_date = plot_df.loc[test_first_valid_index, 'date']
            if test_first_date > self.global_min_date:
                self.global_min_date = test_first_date

            logger.debug(f"{train_last_date} {train_last_valid_index} {test_first_date} {test_first_valid_index}")

        logger.info(f"Global min date: {self.global_min_date}, Global max date: {self.global_max_date}")

        # Collect predictions for the global periods
        self.train_predictions_df_list = []
        self.test_predictions_df_list = []
        self.actual_prices_train = []
        self.actual_prices_test = []
        for ticker in tqdm(self.valid_tickers):
            # Validate ticker data exists
            if ticker not in self.tickers_dict:
                logger.error(f"Ticker {ticker} not found in tickers_dict. Skipping.")
                continue

            if 'metrics_df' not in self.tickers_dict[ticker]:
                logger.error(f"No metrics_df for ticker {ticker}. Skipping.")
                continue

            if 'plot_df' not in self.tickers_dict[ticker]:
                logger.error(f"No plot_df for ticker {ticker}. Skipping.")
                continue

            # Get best model based on lowest Test data MAE (primary metric for log returns)
            metrics_df = self.tickers_dict[ticker]['metrics_df']
            if 'Test data MAE' not in metrics_df.index:
                logger.error(f"'Test data MAE' not found in metrics for {ticker}. Available metrics: {list(metrics_df.index)}")
                continue

            # Get model with lowest MAE (best performance)
            best_model = metrics_df.T['Test data MAE'].idxmin()
            logger.info(f"Best model for {ticker}: {best_model} (MAE: {metrics_df.T.loc[best_model, 'Test data MAE']:.4f})")

            # Generate column names for this model
            train_col = get_column_name(best_model, 'train_predicted_close')
            test_col = get_column_name(best_model, 'test_predicted_close')

            plot_df = self.tickers_dict[ticker]['plot_df']

            # Validate columns exist in plot_df
            required_cols = ['date', 'original_close', train_col, test_col]
            missing_cols = [col for col in required_cols if col not in plot_df.columns]
            if missing_cols:
                logger.error(
                    f"Missing columns {missing_cols} in plot_df for {ticker}. "
                    f"Best model: {best_model}, expected columns: {train_col}, {test_col}. "
                    f"Available columns: {[c for c in plot_df.columns if 'predicted' in c]}. Skipping."
                )
                continue

            # Extract predictions and actual prices
            train_predictions = plot_df[['date', train_col]].copy()
            train_predictions = train_predictions[train_predictions['date'] <= self.global_max_date]
            train_predictions.rename(columns={train_col: ticker}, inplace=True)
            self.train_predictions_df_list.append(train_predictions)

            actual_train = plot_df[['date', 'original_close']].copy()
            actual_train = actual_train[actual_train['date'] <= self.global_max_date]
            actual_train.rename(columns={'original_close': ticker}, inplace=True)
            self.actual_prices_train.append(actual_train)

            test_predictions = plot_df[['date', test_col]].copy()
            test_predictions = test_predictions[test_predictions['date'] >= self.global_min_date]
            test_predictions.rename(columns={test_col: ticker}, inplace=True)
            self.test_predictions_df_list.append(test_predictions)

            actual_test = self.tickers_dict[ticker]['plot_df'][['date', 'original_close']]
            actual_test = actual_test[actual_test['date'] >= self.global_min_date]
            actual_test.rename(columns={'original_close': ticker}, inplace=True)
            self.actual_prices_test.append(actual_test)

        self.selected_features = [self.valid_tickers[0]]
        if not self.valid_tickers:
            raise ValueError("No valid tickers downloaded from Yahoo Finance")

        # Build merged dataframes separately to avoid corrupting loop data
        merged_train_predictions = self.train_predictions_df_list[0].copy()
        merged_actual_train = self.actual_prices_train[0].copy()
        merged_test_predictions = self.test_predictions_df_list[0].copy()
        merged_actual_test = self.actual_prices_test[0].copy()

        #correlation_threshold = 0.9
        for idx, feature in enumerate(self.valid_tickers):
            if idx == 0:
                continue
            logger.debug(f"Processing feature {idx}: {feature}")

            try:
                # Check correlation with already selected features
                tmp = merged_train_predictions.merge(self.train_predictions_df_list[idx], on='date', how='inner')

                if tmp.empty:
                    logger.warning(f"Merge with feature {feature} resulted in empty DataFrame. Skipping.")
                    continue

                correlations = [abs(tmp[feature].corr(tmp[sel_feature])) for sel_feature in self.selected_features]
                logger.debug(f"Correlations: {correlations}")
                max_correlation = max(correlations) if correlations else 0

                # Add feature if correlation is below threshold
                if max_correlation < self.correlation_threshold:
                    # Perform all merges atomically - if any fails, we don't update anything
                    new_train_pred = merged_train_predictions.merge(self.train_predictions_df_list[idx], on='date', how='inner')
                    new_actual_train = merged_actual_train.merge(self.actual_prices_train[idx], on='date', how='inner')
                    new_test_pred = merged_test_predictions.merge(self.test_predictions_df_list[idx], on='date', how='inner')
                    new_actual_test = merged_actual_test.merge(self.actual_prices_test[idx], on='date', how='inner')

                    # Only update if all merges succeeded
                    if new_train_pred.empty or new_actual_train.empty or new_test_pred.empty or new_actual_test.empty:
                        logger.warning(f"One or more merges with feature {feature} resulted in empty DataFrame. Skipping.")
                        continue

                    self.selected_features.append(feature)
                    merged_train_predictions = new_train_pred
                    merged_actual_train = new_actual_train
                    merged_test_predictions = new_test_pred
                    merged_actual_test = new_actual_test

                    logger.info(f"Added feature {feature} (correlation: {max_correlation:.4f})")

            except Exception as e:
                logger.error(f"Error processing feature {feature} at index {idx}: {e}")
                logger.error("Skipping this feature and continuing...")
                continue

        # Update the original lists with merged dataframes only at the end (atomic update)
        self.train_predictions_df_list[0] = merged_train_predictions
        self.actual_prices_train[0] = merged_actual_train
        self.test_predictions_df_list[0] = merged_test_predictions
        self.actual_prices_test[0] = merged_actual_test

        logger.info(f"Selected features: {self.selected_features}")

        selected_features_and_date = ['date'] + self.selected_features
        logger.info(f"Selected features with date: {selected_features_and_date}")

        # Calculate covariance matrix for the training period
        # Predictions are already log returns (preprocess_data outputs log returns)
        train_data = self.train_predictions_df_list[0].drop(columns=['date']).astype(float)
        train_returns = train_data[self.selected_features].dropna()

        self.cov_matrix = train_returns.cov()
        logger.info("Covariance matrix of log-returns for the training period:")
        logger.info(f"\n{self.cov_matrix}")

        # Split the global test period into validation and test sets (50/50 split)
        # Validation: First half of test period (for hyperparameter tuning)
        # Test: Second half of test period (for final evaluation)
        total_test_samples = len(self.test_predictions_df_list[0][selected_features_and_date])
        self.validation_size = int(total_test_samples * 0.5)

        # Ensure we have enough data for both sets
        if self.validation_size < 2:
            raise ValueError(
                f"Test period too small to split. Total samples: {total_test_samples}, "
                f"validation size: {self.validation_size}. Need at least 4 samples total."
            )

        self.validation_data = self.test_predictions_df_list[0][selected_features_and_date].iloc[:self.validation_size]
        self.validation_actual = self.actual_prices_test[0][selected_features_and_date].iloc[:self.validation_size]
        self.test_data = self.test_predictions_df_list[0][selected_features_and_date].iloc[self.validation_size:]
        self.test_actual = self.actual_prices_test[0][selected_features_and_date].iloc[self.validation_size:]

        # Validate no overlap between validation and test sets
        val_dates = set(self.validation_data['date'])
        test_dates = set(self.test_data['date'])
        overlap = val_dates & test_dates
        if overlap:
            raise ValueError(f"Date overlap detected between validation and test sets: {overlap}")

        logger.info(f"Split test period: {total_test_samples} samples -> "
                   f"validation: {len(self.validation_data)}, test: {len(self.test_data)}")

        # Проверка положительной определённости
        if np.any(np.linalg.eigvals(self.cov_matrix) <= 0):
            raise ValueError("Ковариационная матрица не является положительно определённой.")

        return self.cov_matrix, self.validation_data, self.validation_actual, self.test_data, self.test_actual, self.train_predictions_df_list, self.actual_prices_train, self.test_predictions_df_list, self.actual_prices_test, self.tickers_dict

    def refit_and_forecast(self, target_return=None, allow_short=False,
                           clt_significance=None, volatility_z_score=None, ci_alpha=None):
        """
        Refit the best model per ticker on ALL available data, re-run correlation
        filtering on actual log returns, recompute covariance, make a 1-step-ahead
        forecast, optimize portfolio weights, and return buy/sell recommendations.

        Args:
            target_return: Target portfolio return for optimization (optional)
            allow_short: Whether to allow short positions
            clt_significance: CLT significance level α (defaults to config.CLT_SIGNIFICANCE_LEVEL).
                              Converted to z-score internally: z = norm.ppf(1 - α/2).
            volatility_z_score: Volatility z-score threshold (defaults to config.VOLATILITY_Z_SCORE)
            ci_alpha: CI alpha for SARIMA and Chronos (defaults to config.SARIMA_CI_ALPHA)

        Returns:
            List of recommendation dicts with keys:
                ticker, model, predicted_log_return, current_price, limit_price, action, weight
        """
        clt_significance   = clt_significance   if clt_significance   is not None else config.CLT_SIGNIFICANCE_LEVEL
        volatility_z_score = volatility_z_score  if volatility_z_score is not None else config.VOLATILITY_Z_SCORE
        ci_alpha           = ci_alpha           if ci_alpha           is not None else config.SARIMA_CI_ALPHA

        clt_z_score = norm.ppf(1 - clt_significance / 2)

        import torch
        import pandas as pd
        from experiment_runner_for_best_models import refit_for_forecast

        # Step 1: Build actual returns df from already-stored close_stock.
        # Use inner join on Date so only dates present for ALL tickers are kept.
        actual_series = [
            self.tickers_dict[t]['close_stock'].set_index('Date')['Close'].rename(t)
            for t in self.selected_features
        ]
        actual_returns_df = pd.concat(actual_series, axis=1, join='inner')

        # Step 2: Re-run correlation filtering on actual returns
        forward_features = [self.selected_features[0]]
        for ticker in self.selected_features[1:]:
            correlations = [
                abs(actual_returns_df[ticker].corr(actual_returns_df[s]))
                for s in forward_features
            ]
            if max(correlations) < self.correlation_threshold:
                forward_features.append(ticker)
            else:
                logger.info(f"Dropped {ticker} from forward portfolio (actual return correlation too high)")

        logger.info(f"Forward features after correlation filter: {forward_features}")

        # Step 3: Refit best model per ticker (reuses existing close_stock — no re-download)
        refitted = {}
        for ticker in forward_features:
            metrics_df        = self.tickers_dict[ticker]['metrics_df']
            close_stock       = self.tickers_dict[ticker]['close_stock']
            model_hyperparams = self.tickers_dict[ticker]['model_hyperparams']
            best_model_name   = metrics_df.T['Test data MAE'].idxmin()
            hp                = model_hyperparams.get(best_model_name, {})

            logger.info(f"Refitting {best_model_name} for {ticker}...")
            try:
                model, scaler = refit_for_forecast(
                    close_stock=close_stock,
                    model_name=best_model_name,
                    model_hyperparams=hp,
                    time_step_backward=self.time_step_backward
                )
            except Exception as e:
                logger.error(f"Refit failed for {ticker} ({best_model_name}): {e}. Using simulation model.")
                model  = self.tickers_dict[ticker]['models_dict'][best_model_name]
                scaler = self.tickers_dict[ticker]['scaler']

            refitted[ticker] = {
                'model': model,
                'scaler': scaler,
                'best_model_name': best_model_name,
            }

        # Step 4: Make 1-step-ahead prediction with each refitted model
        predicted_returns = {}
        recommendations   = []

        for ticker in forward_features:
            r               = refitted[ticker]
            model           = r['model']
            scaler          = r['scaler']
            best_model_name = r['best_model_name']
            close_stock     = self.tickers_dict[ticker]['close_stock']
            maindf          = self.tickers_dict[ticker]['maindf']

            current_price = float(maindf['Close'].dropna().iloc[-1])
            last_window   = scaler.transform(close_stock[['Close']].values[-self.time_step_backward:])

            try:
                model_ci_ok = True  # default for models without native CI (LSTM, GMDH)

                if best_model_name == 'LSTM':
                    X = last_window.reshape(1, self.time_step_backward, 1)
                    pred_scaled     = model.predict(X, verbose=0)
                    pred_log_return = float(scaler.inverse_transform(pred_scaled)[0][0])

                elif best_model_name == 'SARIMA':
                    pred_scaled, conf_int = model.predict(
                        n_periods=1, return_conf_int=True, alpha=ci_alpha)
                    pred_log_return = float(scaler.inverse_transform(
                        np.array(pred_scaled).reshape(-1, 1))[0][0])
                    ci_lower = float(scaler.inverse_transform(
                        np.array([[conf_int[0][0]]]))[0][0])
                    ci_upper = float(scaler.inverse_transform(
                        np.array([[conf_int[0][1]]]))[0][0])
                    model_ci_ok = not (ci_lower <= 0 <= ci_upper)
                    logger.info(f"  {ticker} SARIMA CI: [{ci_lower:.6f}, {ci_upper:.6f}], "
                                f"contains_zero={not model_ci_ok}")

                elif best_model_name.startswith('GMDH'):
                    X = last_window.reshape(1, -1)
                    pred_scaled     = model.predict(X)
                    pred_log_return = float(scaler.inverse_transform(
                        np.array(pred_scaled).reshape(-1, 1))[0][0])

                elif best_model_name == 'Transformer':
                    X        = torch.tensor(last_window.reshape(1, -1))
                    forecast = model.predict(X, 1,
                                             num_samples=config.TRANSFORMER_CONFIG['num_samples'],
                                             temperature=config.TRANSFORMER_CONFIG['temperature'],
                                             top_k=config.TRANSFORMER_CONFIG['top_k'],
                                             top_p=config.TRANSFORMER_CONFIG['top_p'])
                    samples  = forecast.numpy()
                    alpha    = ci_alpha
                    pred_scaled     = np.quantile(samples, 0.5, axis=1)[:, -1]
                    ci_lower_scaled = np.quantile(samples, alpha / 2, axis=1)[:, -1]
                    ci_upper_scaled = np.quantile(samples, 1 - alpha / 2, axis=1)[:, -1]
                    pred_log_return = float(scaler.inverse_transform(
                        pred_scaled.reshape(-1, 1))[0][0])
                    ci_lower = float(scaler.inverse_transform(
                        ci_lower_scaled.reshape(-1, 1))[0][0])
                    ci_upper = float(scaler.inverse_transform(
                        ci_upper_scaled.reshape(-1, 1))[0][0])
                    model_ci_ok = not (ci_lower <= 0 <= ci_upper)
                    logger.info(f"  {ticker} Chronos CI: [{ci_lower:.6f}, {ci_upper:.6f}], "
                                f"contains_zero={not model_ci_ok}")

                else:
                    logger.warning(f"Unknown model {best_model_name} for {ticker}, skipping.")
                    continue

                # --- Significance checks ---
                metrics_df    = self.tickers_dict[ticker]['metrics_df']
                residual_std  = float(metrics_df.T.loc[best_model_name, 'Test data Residual Std'])
                sigma         = close_stock['Close'].iloc[-config.STOP_LOSS_WINDOW:].std()

                clt_ok = abs(pred_log_return) > clt_z_score * residual_std
                vol_ok = abs(pred_log_return) / (sigma + 1e-10) > volatility_z_score
                is_significant = clt_ok and vol_ok and model_ci_ok

                action = ('BUY' if pred_log_return > 0 else 'SELL') if is_significant else 'HOLD'
                logger.info(f"  {ticker} significance: CLT={clt_ok} "
                            f"(|pred|={abs(pred_log_return):.6f} vs {clt_z_score}×σ_res={clt_z_score*residual_std:.6f}), "
                            f"vol={vol_ok} (z={abs(pred_log_return)/(sigma+1e-10):.2f}), "
                            f"model_ci={model_ci_ok} → {action}")

                # Only include in portfolio optimization if signal is significant
                if is_significant:
                    predicted_returns[ticker] = pred_log_return

                limit_price     = current_price * np.exp(pred_log_return)
                stop_loss_price = current_price * np.exp(-config.STOP_LOSS_SIGMA_MULTIPLIER * sigma)

                recommendations.append({
                    'ticker':               ticker,
                    'model':                best_model_name,
                    'predicted_log_return': round(pred_log_return, 6),
                    'current_price':        round(current_price, 4),
                    'limit_price':          round(limit_price, 4),
                    'stop_loss_price':      round(stop_loss_price, 4),
                    'action':               action,
                    'weight':               0.0,  # filled after optimization
                })
                logger.info(f"  {ticker} {action}: log_return={pred_log_return:.6f}, "
                            f"current={current_price:.4f}, limit={limit_price:.4f}")

            except Exception as e:
                logger.error(f"Forward forecast failed for {ticker} ({best_model_name}): {e}")

        # Step 5: Optimize portfolio weights using actual cov of tickers that have predictions
        tickers_with_pred = [t for t in forward_features if t in predicted_returns]
        if not tickers_with_pred:
            logger.warning("No valid predictions — skipping portfolio optimization.")
            return recommendations

        pred_arr   = np.array([predicted_returns[t] for t in tickers_with_pred])
        cov_matrix = actual_returns_df[tickers_with_pred].cov().values
        if np.any(np.linalg.eigvals(cov_matrix) <= 0):
            cov_matrix = cov_matrix + np.eye(len(tickers_with_pred)) * 1e-8

        portfolio = Portfolio()
        weights   = portfolio.optimize(pred_arr, cov_matrix,
                                       target_return=target_return, allow_short=allow_short)

        weight_map = dict(zip(tickers_with_pred, weights))
        for rec in recommendations:
            rec['weight'] = round(float(weight_map.get(rec['ticker'], 0.0)), 4)

        return recommendations

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
        """
        Process a period for portfolio evaluation using rolling re-optimization.

        At each step T, weights are optimized using predicted returns for T+1,
        then applied to realize actual returns at T+1. This reflects real portfolio
        management where weights are always re-optimized on the latest predictions.

        Both 'data' and 'actual_data' contain log returns (output of preprocess_data).

        Args:
            data: DataFrame with predicted log returns and date column
            actual_data: DataFrame with actual log returns and date column
            cov_matrix: Covariance matrix of returns
            target_return: Target return for optimization (optional)
            allow_short: Whether to allow short positions

        Returns:
            Tuple of (predicted_returns, realized_returns, predicted_volatilities, realized_volatilities)
        """
        realized_returns = []
        predicted_returns = []
        realized_volatilities = []
        predicted_volatilities = []

        for i in range(len(data) - 1):
            current_data = data.iloc[i:i + 2]
            actual_current_data = actual_data.iloc[i:i + 2]

            if not current_data['date'].equals(actual_current_data['date']):
                logger.warning(f"Date mismatch at index {i}: {current_data['date'].values} vs {actual_current_data['date'].values}")

            predicted_return = current_data.drop(columns=['date']).iloc[1]      # predicted log return at T+1
            realized_return = actual_current_data.drop(columns=['date']).iloc[1] # actual log return at T+1

            # Re-optimize weights at every step using predicted returns for T+1
            self.weights = self.optimize(predicted_return, cov_matrix, target_return=target_return,
                                         allow_short=allow_short)

            pred_return, pred_volatility = self.calculate_portfolio_metrics(
                weights=self.weights, returns=predicted_return, cov_matrix=cov_matrix)
            real_return, real_volatility = self.calculate_portfolio_metrics(
                weights=self.weights, returns=realized_return, cov_matrix=cov_matrix)

            realized_returns.append(real_return)
            predicted_returns.append(pred_return)
            realized_volatilities.append(real_volatility)
            predicted_volatilities.append(pred_volatility)

        return predicted_returns, realized_returns, predicted_volatilities, realized_volatilities


    # Calculate accuracy metrics for validation and test sets
    def calculate_accuracy(self, predicted, realized):
        """Calculate normalized mean absolute error (NMAE).

        Returns NMAE if mean of realized is non-zero, otherwise returns MAE.
        """
        mae = np.mean(np.abs(np.array(predicted) - np.array(realized)))
        mean_realized = np.mean(realized)

        # Avoid division by zero or near-zero values
        if abs(mean_realized) < 1e-10:
            logger.warning(f"Mean realized value too close to zero ({mean_realized}), returning MAE instead of normalized error")
            return mae

        return mae / abs(mean_realized)


    # Calculate Sharpe ratio deviation
    def calculate_sharpe_ratio_deviation(self, predicted_returns, realized_returns, predicted_vol, realized_vol):
        """
        Calculate the absolute deviation between predicted and realized Sharpe ratios.

        Sharpe ratio = mean(returns) / mean(volatility)

        Returns:
            Absolute difference between predicted and realized Sharpe ratios, or np.nan if volatility is too low
        """
        mean_pred_vol = np.mean(predicted_vol)
        mean_real_vol = np.mean(realized_vol)

        # Check for near-zero volatility (threshold: 1e-10)
        if abs(mean_pred_vol) < 1e-10 or abs(mean_real_vol) < 1e-10:
            logger.warning(
                f"Volatility too close to zero for Sharpe ratio calculation. "
                f"Predicted vol: {mean_pred_vol}, Realized vol: {mean_real_vol}. "
                f"Returning NaN for Sharpe deviation."
            )
            return np.nan

        predicted_sharpe = np.mean(predicted_returns) / mean_pred_vol
        realized_sharpe = np.mean(realized_returns) / mean_real_vol

        # Check for infinite Sharpe ratios
        if not np.isfinite(predicted_sharpe) or not np.isfinite(realized_sharpe):
            logger.warning(
                f"Non-finite Sharpe ratio detected. "
                f"Predicted: {predicted_sharpe}, Realized: {realized_sharpe}. "
                f"Returning NaN."
            )
            return np.nan

        return abs(predicted_sharpe - realized_sharpe)

    def optimize_portfolio(self, cov_matrix, validation_data, validation_actual, test_data, test_actual, target_return: float | None = None, allow_short: bool = False):

        # Both periods use rolling re-optimization: weights are re-computed at each step T
        # using predicted returns for T+1, then applied to realize actual returns at T+1.
        self.val_pred_returns, self.val_realized_returns, self.val_pred_vol, self.val_realized_vol = self.process_period(
            data=validation_data, actual_data=validation_actual,
            cov_matrix=cov_matrix, target_return=target_return, allow_short=allow_short)

        self.test_pred_returns, self.test_realized_returns, self.test_pred_vol, self.test_realized_vol = self.process_period(
            data=test_data, actual_data=test_actual,
            cov_matrix=cov_matrix, target_return=target_return, allow_short=allow_short)


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

        logger.info(f"Validation Return Accuracy: {self.val_return_accuracy}")
        logger.info(f"Validation Volatility Accuracy: {self.val_volatility_accuracy}")
        logger.info(f"Validation Sharpe Ratio Deviation: {self.val_sharpe_deviation}")
        logger.info(f"Validation Pred Return Sum: {self.val_sum_pred_returns}")
        logger.info(f"Validation Actual Return Sum: {self.val_sum_realized_returns}")

        logger.info(f"Test Return Accuracy: {self.test_return_accuracy}")
        logger.info(f"Test Volatility Accuracy: {self.test_volatility_accuracy}")
        logger.info(f"Test Sharpe Ratio Deviation: {self.test_sharpe_deviation}")
        logger.info(f"Test Pred Return Sum: {self.test_sum_pred_returns}")
        logger.info(f"Test Actual Return Sum: {self.test_sum_realized_returns}")

        #return val_return_accuracy, val_volatility_accuracy, val_sharpe_deviation, np.sum(val_pred_vol), np.sum(val_realized_returns), test_return_accuracy, test_volatility_accuracy, test_sharpe_deviation, np.sum(test_pred_vol), np.sum(test_realized_returns)

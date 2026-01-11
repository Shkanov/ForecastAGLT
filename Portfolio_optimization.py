import streamlit as st

from experiment_runner_for_portfolio import DataLoader, Portfolio
import pickle
from sidebar_portfolio import sidebar


st.set_page_config(
    page_title="Portfolio optimization",
    page_icon="📊")

st.title("Portfolio Optimization")
sidebar_dict = sidebar()
run = st.sidebar.button('Run portfolio optimization')
dataloader = DataLoader()
portfolio = Portfolio()
if run:
    st.header('Price Prediction Results')
    dataloader.experiment_data(top_n = sidebar_dict['top_n'], num_scale_steps = sidebar_dict['num_scale_steps'],
                                    scaling_strategy = sidebar_dict['scaling_strategy'], time_step_backward = sidebar_dict['time_step_backward'])
    #plot_df, metrics_df, models_dict = experiment(ticker = sidebar_dict['ticker'], num_scale_steps= sidebar_dict['num_scale_steps'],
    #           scaling_strategy= sidebar_dict['scaling_strategy'], time_step_backward= sidebar_dict['time_step_backward'])
    col1_tickers, col2_tickers = st.columns(2)
    with col1_tickers:
        st.subheader('Valid Tickers:')
        st.write(dataloader.valid_tickers)
    with col2_tickers:
        st.subheader('Invalid Tickers:')
        st.write(dataloader.invalid_tickers)
    
    col1_date, col2_date = st.columns(2)
    with col1_date:
        st.write('Test Min Date:')
        st.write(dataloader.global_min_date)
    with col2_date:
        st.write('Training Max Date:')
        st.write(dataloader.global_max_date)

    st.subheader('Model Metrics:')
    for ticker in dataloader.valid_tickers:
        st.write(f'{ticker}:')
        st.write('Best model on test data MAPE: ', dataloader.tickers_dict[ticker]['metrics_df'].T.sort_values(by='Test data MAPE', ascending=True).index[0])
        st.write(dataloader.tickers_dict[ticker]['metrics_df'])
        
    st.header('Portfolio Optimization Results')
    portfolio.optimize_portfolio(cov_matrix=dataloader.cov_matrix, validation_data=dataloader.validation_data, validation_actual=dataloader.validation_actual, 
                             test_data=dataloader.test_data, test_actual=dataloader.test_actual, target_return=sidebar_dict['target_return'], allow_short=sidebar_dict['allow_short'])
    col1_weights, col2_weights = st.columns(2)

    with col1_weights:
        st.subheader('Selected tickers:')
        st.write(dataloader.selected_features)
    with col2_weights:
        st.subheader('Portfolio weights:')
        st.write(portfolio.weights)


    col1_results, col2_results = st.columns(2)
    with col1_results:
        st.write(f"Validation Return Accuracy: {portfolio.val_return_accuracy:.4f}")
        st.write(f"Validation Volatility Accuracy: {portfolio.val_volatility_accuracy:.4f}")
        st.write(f"Validation Sharpe Ratio Deviation: {portfolio.val_sharpe_deviation:.4f}")
        st.write(f"Validation Pred Return Sum: {portfolio.val_sum_pred_returns:.4f}")
        st.write(f"Validation Actual Return Sum: {portfolio.val_sum_realized_returns:.4f}")

    with col2_results:
        st.write(f"Test Return Accuracy: {portfolio.test_return_accuracy:.4f}")
        st.write(f"Test Volatility Accuracy: {portfolio.test_volatility_accuracy:.4f}")
        st.write(f"Test Sharpe Ratio Deviation: {portfolio.test_sharpe_deviation:.4f}")
        st.write(f"Test Pred Return Sum: {portfolio.test_sum_pred_returns:.4f}")
        st.write(f"Test Actual Return Sum: {portfolio.test_sum_realized_returns:.4f}")


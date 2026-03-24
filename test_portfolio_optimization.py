"""
Test script to verify portfolio optimization with GMDH on ARM64 macOS
"""
import sys
import traceback
from experiment_runner_for_portfolio import DataLoader, Portfolio

print("="*70)
print("Portfolio Optimization Test with Native GMDH on ARM64 macOS")
print("="*70)

# Test parameters (using small values for quick testing)
test_params = {
    'top_n': 5,  # Use top 5 cryptocurrencies
    'num_scale_steps': 2,  # Reduce from default for faster testing
    'scaling_strategy': 'all',
    'time_step_backward': 30,
    'target_return': 0.02,
    'allow_short': False
}

print("\n📋 Test Parameters:")
for key, value in test_params.items():
    print(f"  {key}: {value}")

try:
    # Initialize
    print("\n" + "="*70)
    print("Step 1: Initializing DataLoader and Portfolio...")
    print("="*70)
    dataloader = DataLoader()
    portfolio = Portfolio()
    print("✅ Initialization successful")

    # Load and prepare data
    print("\n" + "="*70)
    print("Step 2: Loading cryptocurrency data and training models...")
    print("="*70)
    print("⏳ This may take a few minutes as models are being trained...")
    print("   Models: LSTM, SARIMA, GMDH, Chronos Transformer")

    dataloader.experiment_data(
        top_n=test_params['top_n'],
        num_scale_steps=test_params['num_scale_steps'],
        scaling_strategy=test_params['scaling_strategy'],
        time_step_backward=test_params['time_step_backward']
    )

    print("\n✅ Data loading and model training completed!")
    print(f"\n📊 Results:")
    print(f"  Valid tickers: {dataloader.valid_tickers}")
    print(f"  Invalid tickers: {dataloader.invalid_tickers}")
    print(f"  Test date range: {dataloader.global_min_date} to {dataloader.global_max_date}")

    # Show model metrics
    print("\n" + "="*70)
    print("Step 3: Model Performance Metrics")
    print("="*70)
    for ticker in dataloader.valid_tickers:
        metrics_df = dataloader.tickers_dict[ticker]['metrics_df']
        best_model = metrics_df.T.sort_values(by='Test data MAPE', ascending=True).index[0]
        best_mape = metrics_df.T.sort_values(by='Test data MAPE', ascending=True)['Test data MAPE'].iloc[0]
        print(f"\n{ticker}:")
        print(f"  Best model: {best_model}")
        print(f"  Test MAPE: {best_mape:.4f}")
        print("\n  All models performance:")
        print(metrics_df.to_string())

    # Optimize portfolio
    print("\n" + "="*70)
    print("Step 4: Optimizing Portfolio Weights...")
    print("="*70)
    portfolio.optimize_portfolio(
        cov_matrix=dataloader.cov_matrix,
        validation_data=dataloader.validation_data,
        validation_actual=dataloader.validation_actual,
        test_data=dataloader.test_data,
        test_actual=dataloader.test_actual,
        target_return=test_params['target_return'],
        allow_short=test_params['allow_short']
    )

    print("✅ Portfolio optimization completed!")

    # Display results
    print("\n" + "="*70)
    print("Step 5: Portfolio Optimization Results")
    print("="*70)

    print(f"\n📈 Selected Tickers:")
    print(f"  {dataloader.selected_features}")

    print(f"\n💰 Portfolio Weights:")
    for ticker, weight in zip(dataloader.selected_features, portfolio.weights):
        print(f"  {ticker}: {weight:.4f} ({weight*100:.2f}%)")

    print(f"\n📊 Validation Performance:")
    print(f"  Return Accuracy: {portfolio.val_return_accuracy:.4f}")
    print(f"  Volatility Accuracy: {portfolio.val_volatility_accuracy:.4f}")
    print(f"  Sharpe Ratio Deviation: {portfolio.val_sharpe_deviation:.4f}")
    print(f"  Predicted Return Sum: {portfolio.val_sum_pred_returns:.4f}")
    print(f"  Actual Return Sum: {portfolio.val_sum_realized_returns:.4f}")

    print(f"\n📊 Test Performance:")
    print(f"  Return Accuracy: {portfolio.test_return_accuracy:.4f}")
    print(f"  Volatility Accuracy: {portfolio.test_volatility_accuracy:.4f}")
    print(f"  Sharpe Ratio Deviation: {portfolio.test_sharpe_deviation:.4f}")
    print(f"  Predicted Return Sum: {portfolio.test_sum_pred_returns:.4f}")
    print(f"  Actual Return Sum: {portfolio.test_sum_realized_returns:.4f}")

    print("\n" + "="*70)
    print("✅ SUCCESS! Portfolio optimization with GMDH completed successfully!")
    print("="*70)
    print("\n💡 GMDH is fully functional in your cryptocurrency forecasting app!")
    print("🚀 All 4 models (LSTM, SARIMA, GMDH, Chronos) are working natively on ARM64!")

except Exception as e:
    print("\n" + "="*70)
    print("❌ ERROR occurred during portfolio optimization")
    print("="*70)
    print(f"\nError: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

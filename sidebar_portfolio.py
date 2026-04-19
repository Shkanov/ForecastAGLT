

def sidebar():
    import streamlit as st
    import config
    #scale_step_type_list = ['Максимизация доходности при заданном уровне риска','Минимизация риска при заданном уровне доходности']
    #scale_step_type = st.sidebar.selectbox('Оптимизация', scale_step_type_list)
    scaling_strategy_list = ['average_returns', 'median_returns', 'undersampling', 'average_prices', 'median_prices']
    top_n = st.sidebar.number_input('Количество активов-кандидатов', value=5)
    num_scale_steps = st.sidebar.slider('Горизонт инвестирования, дней', 1, 100, 1)
    if num_scale_steps > 1:
        scaling_strategy = st.sidebar.selectbox('Стратегия масштабирования', scaling_strategy_list)
    else:
        scaling_strategy = 'average_returns'  # default, ignored downstream (daily log returns)
    target_return_expander = st.sidebar.expander('Задать целевую доходность')
    use_target_return = target_return_expander.checkbox('Задать целевую доходность')
    if use_target_return:
        target_return = target_return_expander.slider('Уровень доходности, %', 1, 100, 10) * 0.01
    else:
        target_return = None
    time_step_backward = st.sidebar.slider('Количество предикторов, дней', 1, 100, 15)
    allow_short = st.sidebar.checkbox('Разрешить короткие позиции')

    sig_expander = st.sidebar.expander('Signal significance settings')
    clt_significance = sig_expander.slider(
        'CLT significance level (α)', min_value=0.01, max_value=0.50, value=config.CLT_SIGNIFICANCE_LEVEL, step=0.01,
        help=(
            'Controls how strong a predicted log return must be relative to the model\'s own residual noise '
            'before it is treated as an actionable signal.\n\n'
            'Internally converted to a z-score: z = norm.ppf(1 − α/2).\n\n'
            'α=0.01 → z≈2.58 (99% CI, very strict — only large, confident predictions pass)\n'
            'α=0.05 → z≈1.96 (95% CI, recommended default)\n'
            'α=0.10 → z≈1.64 (90% CI, more signals but higher false-positive rate)\n'
            'α=0.20 → z≈1.28 (80% CI, lenient — use only for exploration)\n'
            'α=0.50 → z≈0.67 (50% CI, almost no filtering)\n\n'
            'Lower α = stricter filter = fewer but higher-confidence BUY/SELL signals.'
        ))
    volatility_z_score = sig_expander.slider(
        'Volatility z-score threshold', min_value=0.1, max_value=5.0, value=config.VOLATILITY_Z_SCORE, step=0.1,
        help=(
            'Filters out predictions that are small relative to the asset\'s recent price volatility.\n\n'
            'Computed as: |predicted_log_return| / rolling_σ (σ estimated over last 20 days).\n\n'
            '0.5 → signal must be at least 0.5× the daily σ (very lenient)\n'
            '1.0 → signal must exceed 1 daily σ\n'
            '2.0 → signal must exceed 2× daily σ (recommended default — filters out noise)\n'
            '3.0 → signal must exceed 3× daily σ (very strict, rare signals only)\n\n'
            'Higher threshold = fewer signals, but each one represents a more meaningful predicted move.'
        ))
    ci_alpha = sig_expander.select_slider(
        'Model CI alpha (SARIMA & Chronos)', options=[0.01, 0.05, 0.10, 0.20, 0.50], value=config.SARIMA_CI_ALPHA,
        help=(
            'Significance level for the native confidence intervals produced by SARIMA and Chronos models.\n\n'
            'A signal is blocked if the model\'s own CI contains zero '
            '(i.e. the model itself is uncertain about the direction).\n\n'
            '0.01 → 99% CI (very wide interval — easy to contain zero, strict filter)\n'
            '0.05 → 95% CI (recommended default)\n'
            '0.10 → 90% CI (narrower interval — harder to contain zero, more signals pass)\n'
            '0.20 → 80% CI\n'
            '0.50 → 50% CI (very narrow, almost all signals pass)\n\n'
            'Only applies to SARIMA and Chronos; LSTM and GMDH are unaffected.'
        ))

    return {'top_n': top_n,
            'num_scale_steps': num_scale_steps,
            'scaling_strategy': scaling_strategy,
            'target_return': target_return,
            'time_step_backward': time_step_backward,
            'allow_short': allow_short,
            'clt_significance': clt_significance,
            'volatility_z_score': volatility_z_score,
            'ci_alpha': ci_alpha}


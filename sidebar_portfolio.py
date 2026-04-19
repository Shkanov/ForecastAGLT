

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
    clt_z_score = sig_expander.slider(
        'CLT z-score threshold', min_value=1.0, max_value=3.0, value=config.CLT_Z_SCORE, step=0.01,
        help='|pred| must exceed this many residual std devs to be actionable (~95% CI at 1.96)')
    volatility_z_score = sig_expander.slider(
        'Volatility z-score threshold', min_value=0.5, max_value=4.0, value=config.VOLATILITY_Z_SCORE, step=0.1,
        help='|pred| / rolling_σ must exceed this value')
    ci_alpha = sig_expander.select_slider(
        'Model CI alpha (SARIMA & Chronos)', options=[0.01, 0.05, 0.10], value=config.SARIMA_CI_ALPHA,
        help='Significance level for SARIMA and Chronos native confidence intervals')

    return {'top_n': top_n,
            'num_scale_steps': num_scale_steps,
            'scaling_strategy': scaling_strategy,
            'target_return': target_return,
            'time_step_backward': time_step_backward,
            'allow_short': allow_short,
            'clt_z_score': clt_z_score,
            'volatility_z_score': volatility_z_score,
            'ci_alpha': ci_alpha}


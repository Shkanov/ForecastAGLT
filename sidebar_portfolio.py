

def sidebar():
    import streamlit as st
    #scale_step_type_list = ['Максимизация доходности при заданном уровне риска','Минимизация риска при заданном уровне доходности']
    #scale_step_type = st.sidebar.selectbox('Оптимизация', scale_step_type_list)
    scaling_strategy_list = ['average', 'median', 'undersampling']
    top_n = st.sidebar.number_input('Количество активов-кандидатов', value=5)
    num_scale_steps = st.sidebar.slider('Горизонт инвестирования, дней', 1, 100, 1)
    scaling_strategy = st.sidebar.selectbox('Стратегия масштабирования', scaling_strategy_list)
    target_return_expander = st.sidebar.expander('Задать целевую доходность')
    target_return = target_return_expander.slider('Уровень доходности, %', 1, 100, None)
    if target_return:
        target_return *= 0.01
    time_step_backward = st.sidebar.slider('Количество предикторов, дней', 1, 100, 15)
    allow_short = st.sidebar.checkbox('Разрешить короткие позиции')


    scaling_strategy = 'average'
    time_step_backward = 15
    return {'top_n': top_n,
            'num_scale_steps': num_scale_steps,
            'scaling_strategy': scaling_strategy,
            'target_return': target_return,
            'time_step_backward': time_step_backward, 
            'allow_short': allow_short}


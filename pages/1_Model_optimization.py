import pandas as pd
import numpy as np

# For data preprocessing

# For plotting
import matplotlib.pyplot as plt

import streamlit as st

from gmdh import CriterionType, Criterion, Multi, Combi, Mia, Ria, PolynomialType
from pages.utils.utils import make_prediction_recursive

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

from io import StringIO
import os
os.environ["YF_DISABLE_CURL_CFFI"] = "1"

from logging_config import get_logger
logger = get_logger(__name__)


st.set_page_config(
    page_title="Model optimization",
    page_icon="📈")

@st.cache_resource
def load_chronos_pipeline():
    return get_chronos_pipeline()

pipeline = load_chronos_pipeline()
seed = 42
st.title("Daily price prediction")
tickers = ['BTC', 'ETH', 'BNB', #'USDC',
            'XRP', 'STETH','ADA','DOGE',#'FGC',
           'WTRX','LTC','SOL','TRX','DOT','MATIC','BCH','WBTC','TON11419',
           'DAI','SHIB','AVAX','BUSD','LEO','LINK']
intervals = ['1d']#, '5d', '1wk', '1mo', '3mo'] #['1m', '2m', '5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']
ticker = st.selectbox("Ticker", options=tickers)
interval = st.selectbox("Interval of raw data", options = intervals)

int_to_periods = {'1m':'5d', '2m':'1mo', '5m': '1mo','15m': '1mo','30m': '1mo','60m': '1mo','90m': '1mo',
           '1h': '1y','1d': '10y','5d': '10y','1wk': '10y','1mo': '10y','3mo': '10y'}

period_cut = {'1d': '2022-02-19', '5d': '2020-06-19', '1wk': '2020-06-19', '1mo': '2014-06-19', '3mo': '2014-06-19'}

uploaded_file = st.file_uploader("Choose a file")

# Load data using shared function or from uploaded file
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # To read file as string:
    string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
    maindf = pd.read_csv(uploaded_file)
    maindf = maindf.reset_index() if 'Date' not in maindf.columns else maindf
    maindf['Date'] = pd.to_datetime(maindf['Date'], format='%Y-%m-%d')
    st.write(maindf.head())
else:
    # Use shared data loading function
    maindf = load_crypto_data(ticker, interval)

logger.info(f'Total number of days present in the dataset: {maindf.shape[0]}')
logger.info(f'Total number of fields present in the dataset: {maindf.shape[1]}')
logger.debug(f'Dataset head:\n{maindf.head()}')

y_overall = maindf.copy()#.loc[(maindf['Date'] >= '2014-09-17')]
                     #& (maindf['Date'] <= '2022-02-19')]

global_expander = st.sidebar.expander('Параметры режима моделирования')
scaling_expander= st.sidebar.expander('Режим масштабирования')
scaling_strategy_list = ['average_returns', 'median_returns', 'undersampling', 'average_prices', 'median_prices']
scale_step_type_list = ['D','W','M','Y']
scale_step_type = scaling_expander.selectbox('Шаг масштабирования', scale_step_type_list)
num_scale_steps = scaling_expander.slider('Размер шага масштабирования', 1, 100, 1)

# Use shared preprocessing function
if num_scale_steps > 1:
    scaling_strategy = scaling_expander.selectbox('Метод масштабирования', scaling_strategy_list)
else:
    scaling_strategy = 'average_returns'  # Default, won't be used when num_scale_steps=1

y_overall = preprocess_data(y_overall, num_scale_steps, scaling_strategy, scale_step_type)


#names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])
fig, ax = plt.subplots()
#ax.plot(y_overall.Date, y_overall['Close'], label = 'Stock Close Price')
ax.plot(y_overall['Close'], label='Log Return')

ax.legend()
ax.set_ylabel("Log Return")
ax.set_title(f'Log return dynamics for {ticker}')

#st.image(fig)
st.pyplot(fig)
plt.close(fig)  # Explicitly close to prevent memory leaks
#fig.show()




train = st.sidebar.button('Train')
time_step_backward = st.sidebar.slider('Количество шагов назад для предикторов', 5, 60, 15)
time_step_forward = st.sidebar.slider('Количество шагов вперед для таргета', 1, 60, 1)


pred_days = 1
recursive_pred = False
if time_step_forward == 1:
    expander = st.sidebar.expander('Режим ресурсивного прогноза')
    pred_days = expander.slider('Количество шагов для ресурсивного прогноза', 1, 30, 15)
    recursive_pred = expander.checkbox('Запустить рекурсивный прогноз')



GMDH = st.sidebar.checkbox('Добавить режим МГУА')
transformer = st.sidebar.checkbox('Добавить режим Transformer')
if GMDH:
    expander1 = st.sidebar.expander('Гиперпараметры МГУА')
    GMDHs = {'Combi': Combi(), 'Multi': Multi(), 'Mia': Mia(), 'Ria': Ria()}
    criterions = {'Критерий регулярности (несимметричная форма)': CriterionType.REGULARITY,
                  'Критерий регулярности (симметричная форма)': CriterionType.SYM_REGULARITY,
                  'Критерий стабильности (несимметричная форма)': CriterionType.STABILITY,
                  'Критерий стабильности (симметричная форма)': CriterionType.SYM_STABILITY,
                  'Критерий минимума смещения коэффициентов': CriterionType.UNBIASED_COEFFS,
                  'Критерий минимума смещения решений (несимметричная форма)': CriterionType.UNBIASED_OUTPUTS,
                  'Критерий минимума смещения решений (симметричная форма)': CriterionType.SYM_UNBIASED_OUTPUTS,
                  'Абсолютно помехоустойчивый критерий (несимметричная форма)': CriterionType.ABSOLUTE_NOISE_IMMUNITY,
                  'Абсолютно помехоустойчивый критерий (симметричная форма)': CriterionType.SYM_ABSOLUTE_NOISE_IMMUNITY}
    polynoms = {'LINEAR': PolynomialType.LINEAR,
                  'LINEAR_COV': PolynomialType.LINEAR_COV,
                  'QUADRATIC': PolynomialType.QUADRATIC}
    GMDH_algo = expander1.selectbox("Алгоритм МГУА", options = GMDHs.keys())
    criterion = expander1.selectbox("Внешний критерий", options = criterions.keys())
    p_average = expander1.slider('p_average', 1, 10, 1)
    limit = expander1.number_input('limit', value = 0.)
    k_best = expander1.slider('k_best', 1, 10, 3 if GMDH_algo == 'Mia' else 1)
    polynom = expander1.selectbox("Вид базовых полиномов", options = polynoms.keys())


# Check if columns have MultiIndex before dropping level
if isinstance(y_overall.columns, pd.MultiIndex):
    y_overall.columns = y_overall.columns.droplevel(1)
#y_overall = y_overall.reset_index()


if train:
    my_bar = st.progress(0, text='Model training progress. Preparing data')
    # Prepare data using shared function
    closedf = y_overall[['Date', 'Close']].dropna()
    logger.info(f"Shape of close dataframe: {closedf.shape}")

    my_bar.progress(10 + 1, text='Prepared data -> Splitting and scaling')

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

    my_bar.progress(30 + 1, text='Data prepared -> Reshaping for models')

    # Keep GMDH copy before reshaping for LSTM
    X_train_gmdh = X_train.copy()
    X_test_gmdh = X_test.copy()

    # Get scaled data for recursive predictions using the same scaler as the main pipeline
    # (close_stock already has ≤1000 samples from prepare_train_test_data)
    closedf_for_recursive = close_stock[['Close']]
    training_size = int(len(closedf_for_recursive) * 0.70)
    train_data = scaler.transform(closedf_for_recursive[:training_size])
    test_data = scaler.transform(closedf_for_recursive[training_size:])

    my_bar.progress(40 + 1, text='Data prepared -> Training LSTM model')

    # Train LSTM using shared function
    lstm_model, history = train_lstm(
        X_train, y_train, X_test, y_test,
        seed=seed,
        lstm_units=10,
        epochs=100,
        batch_size=32,
        patience=30,
        verbose=False
    )

    my_bar.progress(55 + 1, text='Trained LSTM -> Training SARIMA model')

    # Train SARIMA using shared function
    arima_model = train_sarima(
        train_data,
        time_step_backward=time_step_backward,
        seasonal=True,
        m=12,
        trace=True
    )
    st.text(arima_model.summary())

    if GMDH:
        gmdh_params = {
            'algorithms': {
                'GMDH': {
                    'type': GMDH_algo,
                    'criterion': criterions[criterion],
                    'p_average': p_average,
                    'limit': limit,
                    'k_best': k_best,
                    'polynomial_type': polynoms[polynom]
                }
            }
        }
        gmdh_models = train_gmdh_models(X_train_gmdh, y_train, gmdh_params)
        model_gmdh = gmdh_models['GMDH']
        st.write(f"GMDH model: {model_gmdh.get_best_polynomial()}")


    # Build models dictionary
    models_dict = {'LSTM': lstm_model, 'SARIMA': arima_model}
    if GMDH:
        models_dict['GMDH'] = model_gmdh
    if transformer:
        models_dict['Transformer'] = pipeline

    my_bar.progress(70 + 1, text='Trained models -> Calculating loss')
    import matplotlib.pyplot as plt

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    fig, ax = plt.subplots()
    ax.plot(epochs, loss, 'r', label='Training loss')
    ax.plot(epochs, val_loss, 'b', label='Validation loss')
    ax.legend()
    ax.set_title('Потери на обучении и валидации')
    #ax.set_ylim[0, 0.2]
    st.pyplot(fig)
    plt.close(fig)  # Explicitly close to prevent memory leaks

    my_bar.progress(80 + 1, text='Calculated loss -> Making predictions')

    # Make predictions using shared function
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

    my_bar.progress(85 + 1, text='Made predictions -> Calculating performance metrics')

    # Calculate metrics using shared function
    metrics_df = calculate_all_metrics(y_train, y_test, predictions, scaler)
    st.write(metrics_df)
    #print("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
    #print("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
    #print("----------------------------------------------------------------------")
    #print("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
    #print("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))


    my_bar.progress(90 + 1, text='Calculated performance metrics -> Creating plot dataframe')

    # Create plot dataframe using shared function
    plotdf = create_plot_dataframe(
        close_stock=close_stock,
        predictions=predictions,
        time_step_backward=time_step_backward,
        time_step_forward=time_step_forward
    )

    # Create comparison plot
    fig, ax = plt.subplots()
    ax.plot(plotdf['date'], plotdf['original_close'], label='Actual log return')

    # Plot predictions for each model
    for model_name in predictions.keys():
        model_suffix = model_name.lower().replace(' ', '_')
        if f'train_predicted_close_{model_suffix}' in plotdf.columns:
            ax.plot(plotdf['date'], plotdf[f'train_predicted_close_{model_suffix}'],
                    label=f'Train predicted log return ({model_name})')
            ax.plot(plotdf['date'], plotdf[f'test_predicted_close_{model_suffix}'],
                    label=f'Test predicted log return ({model_name})')

    ax.legend()
    ax.set_title("Сравнение исходных и смоделированных лог-доходностей")
    ax.set_ylabel("Log Return")
    st.pyplot(fig)
    plt.close(fig)  # Explicitly close to prevent memory leaks

    my_bar.progress(100, text='Done')


    if recursive_pred:
        lst_output_arima = make_prediction_recursive(test_data=test_data, method='SARIMA', model=arima_model,
                                                     scaler=scaler, pred_days=pred_days,
                                                     time_step_backward=time_step_backward)
        lst_output_lstm = make_prediction_recursive(test_data=test_data, method='LSTM', model=lstm_model,
                                                    scaler=scaler, pred_days=pred_days,
                                                    time_step_backward=time_step_backward)
        if GMDH:
            lst_output_gmdh = make_prediction_recursive(test_data=test_data, method='GMDH', model=model_gmdh,
                                                        scaler=scaler, pred_days=pred_days,
                                                        time_step_backward=time_step_backward)
        if transformer:
            lst_output_transformer = make_prediction_recursive(test_data=test_data, method='Transformer', model=pipeline,
                                                               scaler=scaler, pred_days=pred_days,
                                                               time_step_backward=time_step_backward)

        last_days = np.arange(1, time_step_backward + 1)
        day_pred = np.arange(time_step_backward + 1, time_step_backward + pred_days + 1)
        logger.debug(f"Last days: {last_days}")
        logger.debug(f"Day pred: {day_pred}")

        temp_mat = np.empty((len(last_days) + pred_days, 1))
        temp_mat[:] = np.nan

        last_original_days_value = temp_mat.copy()
        next_predicted_days_value_arima = temp_mat.copy()
        next_predicted_days_value_lstm = temp_mat.copy()
        if GMDH:
            next_predicted_days_value_gmdh = temp_mat.copy()
        if transformer:
            next_predicted_days_value_transformer = temp_mat.copy()

        last_original_days_value[0:time_step_backward] = \
            closedf_for_recursive[len(closedf_for_recursive) - time_step_backward:].values
        next_predicted_days_value_arima[time_step_backward:] = lst_output_arima
        next_predicted_days_value_lstm[time_step_backward:] = lst_output_lstm
        if GMDH:
            next_predicted_days_value_gmdh[time_step_backward:] = lst_output_gmdh
        if transformer:
            next_predicted_days_value_transformer[time_step_backward:] = lst_output_transformer

        if GMDH:
            if transformer:
                new_pred_plot = pd.DataFrame({
                    'last_original_days_value': last_original_days_value.reshape(1, -1).tolist()[0],
                    'next_predicted_days_value_arima': next_predicted_days_value_arima.reshape(1, -1).tolist()[0],
                    'next_predicted_days_value_lstm': next_predicted_days_value_lstm.reshape(1, -1).tolist()[0],
                    'next_predicted_days_value_gmdh': next_predicted_days_value_gmdh.reshape(1, -1).tolist()[0],
                    'next_predicted_days_value_transformer':
                        next_predicted_days_value_transformer.reshape(1, -1).tolist()[0]
                })
            elif not transformer:
                new_pred_plot = pd.DataFrame({
                    'last_original_days_value': last_original_days_value.reshape(1, -1).tolist()[0],
                    'next_predicted_days_value_arima': next_predicted_days_value_arima.reshape(1, -1).tolist()[0],
                    'next_predicted_days_value_lstm': next_predicted_days_value_lstm.reshape(1, -1).tolist()[0],
                    'next_predicted_days_value_gmdh': next_predicted_days_value_gmdh.reshape(1, -1).tolist()[0]
                })
        elif not GMDH:
            if transformer:
                new_pred_plot = pd.DataFrame({
                    'last_original_days_value': last_original_days_value.reshape(1, -1).tolist()[0],
                    'next_predicted_days_value_arima': next_predicted_days_value_arima.reshape(1, -1).tolist()[0],
                    'next_predicted_days_value_lstm': next_predicted_days_value_lstm.reshape(1, -1).tolist()[0],
                    'next_predicted_days_value_transformer':
                        next_predicted_days_value_transformer.reshape(1, -1).tolist()[0]
                })
            else:
                new_pred_plot = pd.DataFrame({
                    'last_original_days_value': last_original_days_value.reshape(1, -1).tolist()[0],
                    'next_predicted_days_value_arima': next_predicted_days_value_arima.reshape(1, -1).tolist()[0],
                    'next_predicted_days_value_lstm': next_predicted_days_value_lstm.reshape(1, -1).tolist()[0]
                })
        fig, ax = plt.subplots()
        ax.plot(new_pred_plot.index, new_pred_plot['last_original_days_value'],
                label=f"Last {time_step_backward} steps (actual log return)")
        ax.plot(new_pred_plot.index, new_pred_plot['next_predicted_days_value_arima'],
                label=f"Next {pred_days} predicted log returns (SARIMA)")
        ax.plot(new_pred_plot.index, new_pred_plot['next_predicted_days_value_lstm'],
                label=f"Next {pred_days} predicted log returns (LSTM)")
        if GMDH:
            ax.plot(new_pred_plot.index, new_pred_plot['next_predicted_days_value_gmdh'],
                    label=f"Next {pred_days} predicted log returns (GMDH)")
        if transformer:
            ax.plot(new_pred_plot.index, new_pred_plot['next_predicted_days_value_transformer'],
                    label=f"Next {pred_days} predicted log returns (Transformer)")
        ax.legend()
        ax.set_ylabel("Log Return")
        ax.set_title(f"Last {time_step_backward} actual vs next {pred_days} predicted log returns")
        # Auto-scale: log returns include negatives so do not force ylim to start at 0
        st.pyplot(fig)
        plt.close(fig)  # Explicitly close to prevent memory leaks
        #ax.plot()


    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode("utf-8")
    @st.cache_data
    def convert_metrics_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode("utf-8")

    plotdf_csv = convert_df(plotdf)
    metrics_df_csv = convert_metrics_df(metrics_df)
    st.download_button('Download data', plotdf_csv, file_name='predictions.csv', mime="text/csv")
    st.download_button('Download metrics', metrics_df_csv, file_name='metrics.csv', mime="text/csv")



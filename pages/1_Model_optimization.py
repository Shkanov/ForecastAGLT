import pandas as pd
import numpy as np
import math

# For Evalution we will use these library

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# For model building we will use these library

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping

# For PLotting we will use these library
import matplotlib.pyplot as plt

import yfinance as yf

import streamlit as st

from gmdh import CriterionType, Criterion, Multi, Combi, Mia, Ria, PolynomialType
from chronos import ChronosPipeline
import torch
import pmdarima as pm
from pages.utils.utils import create_dataset, make_prediction, make_prediction_recursive

from io import StringIO

st.set_page_config(
    page_title="Model optimization",
    page_icon="📈")

@st.cache_data
def get_pipeline():
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",  # use "cpu" for CPU inference and "mps" for Apple Silicon
        torch_dtype=torch.bfloat16)
    return pipeline

pipeline = get_pipeline()
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


try:
    maindf = yf.download(tickers = f"{ticker}-USD",  # list of tickers
                period = int_to_periods[interval],         # time period
                interval = interval,       # trading interval
                prepost = False,       # download pre/post market hours data?
                repair = True,)         # repair obvious price errors e.g. 100x?
    if len(maindf) == 0:
        raise FileNotFoundError
except:
    maindf = pd.read_csv(f'{ticker}.csv')

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # To read file as string:
    string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
    maindf = pd.read_csv(uploaded_file)
    st.write(maindf.head())



maindf=maindf.reset_index()
maindf['Date'] = pd.to_datetime(maindf['Date'], format='%Y-%m-%d')

#maindf = pd.read_csv('BTC-USD.csv')
print('Total number of days present in the dataset: ',maindf.shape[0])
print('Total number of fields present in the dataset: ',maindf.shape[1])
print(maindf.head())

y_overall = maindf.copy()#.loc[(maindf['Date'] >= '2014-09-17')]
                     #& (maindf['Date'] <= '2022-02-19')]

global_expander = st.sidebar.expander('Параметры режима моделирования')
scaling_expander= st.sidebar.expander('Режим масштабирования')
scaling_strategy_list = ['median', 'average', 'undersampling']
scale_step_type_list = ['D','W','M','Y']
scale_step_type = scaling_expander.selectbox('Шаг масштабирования', scale_step_type_list)
num_scale_steps = scaling_expander.slider('Размер шага масштабирования', 1, 100, 1)

y_overall = y_overall[['Date','Close']]
if num_scale_steps > 1:
    scaling_strategy = scaling_expander.selectbox('Метод масштабирования', scaling_strategy_list)
    scaling_step_combined = str(num_scale_steps) + scale_step_type
    # Определяем сегодняшнюю дату
    today = pd.Timestamp.now().normalize()
    if scaling_strategy == 'average':
        # y_overall = y_overall.groupby(pd.Grouper(key = 'Date', freq = scaling_step_combined)).mean()
        # Добавляем колонку для конца интервала
        y_overall['Interval_End'] = today - (
                (today - y_overall['Date']) // pd.Timedelta(scaling_step_combined)) * pd.Timedelta(
            scaling_step_combined)
        # Группируем по интервалам и считаем среднее
        y_overall = y_overall.groupby('Interval_End')['Close'].mean().reset_index()
        # Сортируем результат
        y_overall = y_overall.sort_values('Interval_End')  # .reset_index(drop=True)
        y_overall = y_overall.rename({'Interval_End': 'Date'}, axis=1)
    elif scaling_strategy == 'median':
        # y_overall = y_overall.groupby(pd.Grouper(key = 'Date', freq = scaling_step_combined)).median()
        # y_overall = y_overall.groupby(pd.Grouper(key = 'Date', freq = scaling_step_combined)).mean()
        # Добавляем колонку для конца интервала
        y_overall['Interval_End'] = today - (
                (today - y_overall['Date']) // pd.Timedelta(scaling_step_combined)) * pd.Timedelta(
            scaling_step_combined)
        # Группируем по интервалам и считаем среднее
        y_overall = y_overall.groupby('Interval_End')['Close'].median().reset_index()
        # Сортируем результат
        y_overall = y_overall.sort_values('Interval_End')  # .reset_index(drop=True)
        y_overall = y_overall.rename({'Interval_End': 'Date'}, axis=1)
    else:
        # y_overall = y_overall.resample(on = 'Date', rule = scaling_step_combined).last()
        # Устанавливаем 'Date' как индекс, если это ещё не сделано
        # y_overall = y_overall.set_index('Date')
        # y_overall.columns = y_overall.columns.droplevel(1)
        y_overall = y_overall.resample(on='Date', rule=scaling_step_combined, origin='end').last()
        y_overall = y_overall.reset_index()


#names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])
fig, ax = plt.subplots()
#ax.plot(y_overall.Date, y_overall['Close'], label = 'Stock Close Price')
ax.plot(y_overall['Close'], label = 'Stock Close Price')

ax.legend()
ax.set_title(f'Динамика цены закрытия для {ticker}')

#st.image(fig)
st.pyplot(fig)
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


y_overall.columns = y_overall.columns.droplevel(1)#.droplevel()
#y_overall = y_overall.reset_index()


if train:
    my_bar = st.progress(0, text='Model training progress. Truncating the dataset now')
    # Lets First Take all the Close Price
    closedf = y_overall[['Date', 'Close']]#maindf[['Date', 'Close']]
    print("Shape of close dataframe:", closedf.shape)
    closedf = closedf[-1000:]#closedf[closedf['Date'] > period_cut[interval]]
    close_stock = closedf.copy()
    print("Total data for prediction: ", closedf.shape[0])
    my_bar.progress(10 + 1, text='Truncated the dataset -> Scaling it')
    # deleting date column and normalizing using MinMax Scaler


    del closedf['Date']
    scaler = MinMaxScaler(feature_range=(0, 1))
    #closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))
    print(closedf.shape)

    my_bar.progress(20 + 1, text='Scaled the dataset -> Splitting it into subsamples')
    # we keep the training set as 60% and 40% testing set

    training_size = int(len(closedf) * 0.70)
    test_size = len(closedf) - training_size
    assert test_size > time_step_backward + time_step_forward, "Test_size is shorter than time_step_backward + time_step_forward"
    train_data, test_data = closedf[0:training_size], closedf[training_size:len(closedf)]
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    print("train_data: ", train_data.shape)
    print("test_data: ", test_data.shape)

    my_bar.progress(30 + 1, text='Split it into subsamples -> Cutting them into observations')

    X_train, y_train = create_dataset(train_data, time_step_backward, time_step_forward)
    X_test, y_test = create_dataset(test_data, time_step_backward, time_step_forward)

    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_test: ", X_test.shape)
    print("y_test", y_test.shape)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train_gmdh = X_train.copy()
    X_test_gmdh = X_test.copy()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print("X_train: ", X_train.shape)
    print("X_test: ", X_test.shape)

    my_bar.progress(40 + 1, text='Cut it into observations -> Training the model')
    model = Sequential()
    model.add(LSTM(10, input_shape=(None, 1), activation="relu",
                   kernel_initializer = initializers.GlorotNormal(seed = seed), bias_initializer = initializers.GlorotNormal(seed = seed)))
    model.add(Dense(1,
                   kernel_initializer = initializers.GlorotNormal(seed = seed), bias_initializer = initializers.GlorotNormal(seed = seed)))
    model.compile(loss="mean_squared_error", optimizer="adam")
    callback = EarlyStopping(monitor='loss', patience=30, restore_best_weights = True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=10,
                        callbacks = [callback])

    arima_model = pm.auto_arima(train_data,
                          m=12,  # frequency of series
                          seasonal=True,  # TRUE if seasonal series
                          d=None,  # let model determine 'd'
                          test='adf',  # use adftest to find optimal 'd'
                          start_p=0, start_q=0,  # minimum p and q
                          max_p=time_step_backward, max_q=time_step_backward,  # maximum p and q
                          D=None,  # let model determine 'D'
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    st.text(arima_model.summary())

    if GMDH:
        model_gmdh = GMDHs[GMDH_algo]
        if GMDH_algo == 'Combi':
            model_gmdh.fit(X_train_gmdh, y_train, p_average = p_average, limit = limit, test_size=0.3,
                           criterion = Criterion(criterion_type = criterions[criterion]))
        if GMDH_algo == 'Multi':
            model_gmdh.fit(X_train_gmdh, y_train, p_average=p_average, limit=limit, test_size=0.3,
                           criterion=Criterion(criterion_type=criterions[criterion]),
                            k_best = k_best)
        if GMDH_algo in ['Ria', 'Mia']:
            model_gmdh.fit(X_train_gmdh, y_train, p_average=p_average, limit=limit, test_size=0.3,
                           criterion=Criterion(criterion_type=criterions[criterion]),
                            k_best = k_best, polynomial_type = polynoms[polynom])
        st.write(f"GMDH model: {model_gmdh.get_best_polynomial()}")


    my_bar.progress(70 + 1, text='Trained model -> Calculating loss')
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

    my_bar.progress(80 + 1, text='Calculated loss -> Scoring the dataset')


    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    train_predict, test_predict = make_prediction(X_train, X_test, method='LSTM', model=model,
                    scaler=scaler, time_step_forward=time_step_forward)
    train_predict_arima, test_predict_arima = make_prediction(X_train, X_test, method='SARIMA', model=arima_model,
                    scaler=scaler, time_step_forward=time_step_forward)
    if GMDH:
        train_predict_gmdh, test_predict_gmdh = make_prediction(X_train_gmdh, X_test_gmdh, method='GMDH', model=model_gmdh,
                        scaler=scaler, time_step_forward=time_step_forward)
    if transformer:
        X_train_forecast_median, X_test_forecast_median = make_prediction(X_train_gmdh, X_test_gmdh, method='Transformer', model=pipeline,
                        scaler=scaler, time_step_forward=time_step_forward)

    my_bar.progress(85 + 1, text='Scored the dataset -> Calculating perfomance metrics')

    # Evaluation metrices RMSE and MAE
    metrics_tmp = {}
    metrics1 = {}
    metrics1['LSTM'] = []
    #metrics1['Transformer'] = []
    metrics_tmp["Train data RMSE"] = math.sqrt(mean_squared_error(original_ytrain, train_predict))
    metrics_tmp["Train data MSE"] = mean_squared_error(original_ytrain, train_predict)
    metrics_tmp["Train data MAE"] =  mean_absolute_error(original_ytrain, train_predict)
    print("-------------------------------------------------------------------------------------")
    metrics_tmp["Test data RMSE"] =  math.sqrt(mean_squared_error(original_ytest, test_predict))
    metrics_tmp["Test data MSE"] =  mean_squared_error(original_ytest, test_predict)
    metrics_tmp["Test data MAE"] =  mean_absolute_error(original_ytest, test_predict)
    #metrics_tmp["Train data explained variance regression score"] = explained_variance_score(original_ytrain, train_predict)
    #metrics_tmp["Test data explained variance regression score"] = explained_variance_score(original_ytest, test_predict)
    metrics_tmp["Train data R2 score"] =  r2_score(original_ytrain, train_predict)
    metrics_tmp["Test data R2 score"] =  r2_score(original_ytest, test_predict)
    for metric in metrics_tmp:
        print(metric, ': ', metrics_tmp[metric])
        metrics1['LSTM'].append(metrics_tmp[metric])


    metrics1['SARIMA'] = []
    # metrics1['Transformer'] = []
    metrics_tmp["Train data RMSE"] = math.sqrt(mean_squared_error(original_ytrain, train_predict_arima))
    metrics_tmp["Train data MSE"] = mean_squared_error(original_ytrain, train_predict_arima)
    metrics_tmp["Train data MAE"] = mean_absolute_error(original_ytrain, train_predict_arima)
    print("-------------------------------------------------------------------------------------")
    metrics_tmp["Test data RMSE"] = math.sqrt(mean_squared_error(original_ytest, test_predict_arima))
    metrics_tmp["Test data MSE"] = mean_squared_error(original_ytest, test_predict_arima)
    metrics_tmp["Test data MAE"] = mean_absolute_error(original_ytest, test_predict_arima)
    # metrics_tmp["Train data explained variance regression score"] = explained_variance_score(original_ytrain, train_predict)
    # metrics_tmp["Test data explained variance regression score"] = explained_variance_score(original_ytest, test_predict)
    metrics_tmp["Train data R2 score"] = r2_score(original_ytrain, train_predict_arima)
    metrics_tmp["Test data R2 score"] = r2_score(original_ytest, test_predict_arima)
    for metric in metrics_tmp:
        print(metric, ': ', metrics_tmp[metric])
        metrics1['SARIMA'].append(metrics_tmp[metric])
    if GMDH:
        metrics1['GMDH'] = []
        metrics_tmp["Train data RMSE"] = math.sqrt(mean_squared_error(original_ytrain, train_predict_gmdh))
        metrics_tmp["Train data MSE"] = mean_squared_error(original_ytrain, train_predict_gmdh)
        metrics_tmp["Train data MAE"] =  mean_absolute_error(original_ytrain, train_predict_gmdh)
        print("-------------------------------------------------------------------------------------")
        metrics_tmp["Test data RMSE"] =  math.sqrt(mean_squared_error(original_ytest, test_predict_gmdh))
        metrics_tmp["Test data MSE"] =  mean_squared_error(original_ytest, test_predict_gmdh)
        metrics_tmp["Test data MAE"] =  mean_absolute_error(original_ytest, test_predict_gmdh)
        #metrics_tmp["Train data explained variance regression score"] = explained_variance_score(original_ytrain, train_predict)
        #metrics_tmp["Test data explained variance regression score"] = explained_variance_score(original_ytest, test_predict)
        metrics_tmp["Train data R2 score"] =  r2_score(original_ytrain, train_predict_gmdh)
        metrics_tmp["Test data R2 score"] =  r2_score(original_ytest, test_predict_gmdh)
        for metric in metrics_tmp:
            print(metric, ': ', metrics_tmp[metric])
            metrics1['GMDH'].append(metrics_tmp[metric])

    if transformer:
        metrics1['Transformer'] = []
        metrics_tmp["Train data RMSE"] = math.sqrt(mean_squared_error(original_ytrain, X_train_forecast_median))
        metrics_tmp["Train data MSE"] = mean_squared_error(original_ytrain, X_train_forecast_median)
        metrics_tmp["Train data MAE"] = mean_absolute_error(original_ytrain, X_train_forecast_median)
        print("-------------------------------------------------------------------------------------")
        metrics_tmp["Test data RMSE"] = math.sqrt(mean_squared_error(original_ytest, X_test_forecast_median))
        metrics_tmp["Test data MSE"] = mean_squared_error(original_ytest, X_test_forecast_median)
        metrics_tmp["Test data MAE"] = mean_absolute_error(original_ytest, X_test_forecast_median)
        # metrics_tmp["Train data explained variance regression score"] = explained_variance_score(original_ytrain, train_predict)
        # metrics_tmp["Test data explained variance regression score"] = explained_variance_score(original_ytest, test_predict)
        metrics_tmp["Train data R2 score"] = r2_score(original_ytrain, X_train_forecast_median)
        metrics_tmp["Test data R2 score"] = r2_score(original_ytest, X_test_forecast_median)
        for metric in metrics_tmp:
            print(metric, ': ', metrics_tmp[metric])
            metrics1['Transformer'].append(metrics_tmp[metric])

    metrics_df = pd.DataFrame.from_dict(metrics1, orient = 'columns')#(metrics, columns = ['LSTM', 'GMDH'])
    metrics_df.index = metrics_tmp.keys()
    st.write(metrics_df)
    #print("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
    #print("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
    #print("----------------------------------------------------------------------")
    #print("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
    #print("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))


    my_bar.progress(90 + 1, text='Calculated performance metrics -> Plotting predictions')

    # shift train predictions for plotting

    lag = time_step_backward + (time_step_forward - 1)
    trainPredictPlot_arima = np.empty_like(closedf)
    trainPredictPlot_arima[:, :] = np.nan
    trainPredictPlot_arima[lag:len(train_predict_arima) + lag, :] = train_predict_arima
    print(trainPredictPlot_arima[lag:len(train_predict_arima) + lag, :].shape, train_predict_arima.shape)
    print("Train predicted data: ", trainPredictPlot_arima.shape)

    # shift test predictions for plotting
    testPredictPlot_arima = np.empty_like(closedf)
    testPredictPlot_arima[:, :] = np.nan
    testPredictPlot_arima[len(train_predict_arima) + (lag * 2):len(closedf), :] = test_predict_arima
    print(testPredictPlot_arima[len(train_predict_arima) + (lag * 2):len(closedf), :].shape, test_predict_arima.shape)
    print("Test predicted data: ", testPredictPlot_arima.shape)



    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lag:len(train_predict) + lag, :] = train_predict
    print(trainPredictPlot[lag:len(train_predict) + lag, :].shape, train_predict.shape)
    print("Train predicted data: ", trainPredictPlot.shape)

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (lag * 2):len(closedf), :] = test_predict
    print(testPredictPlot[len(train_predict) + (lag * 2):len(closedf), :].shape, test_predict.shape)
    print("Test predicted data: ", testPredictPlot.shape)

    if GMDH:
        trainPredictPlot_gmdh = np.empty_like(closedf)
        trainPredictPlot_gmdh[:, :] = np.nan
        trainPredictPlot_gmdh[lag:len(train_predict_gmdh) + lag, :] = train_predict_gmdh
        print(trainPredictPlot_gmdh[lag:len(train_predict_gmdh) + lag, :].shape, train_predict_gmdh.shape)

        testPredictPlot_gmdh = np.empty_like(closedf)
        testPredictPlot_gmdh[:, :] = np.nan
        testPredictPlot_gmdh[len(train_predict_gmdh) + (lag * 2):len(closedf), :] = test_predict_gmdh
        print(testPredictPlot_gmdh[len(train_predict_gmdh) + (lag * 2):len(closedf), :].shape, test_predict_gmdh.shape)

    if transformer:
        trainPredictPlot_transformer = np.empty_like(closedf)
        trainPredictPlot_transformer[:, :] = np.nan
        trainPredictPlot_transformer[lag:len(X_train_forecast_median) + lag, :] = X_train_forecast_median
        print(trainPredictPlot_transformer[lag:len(X_train_forecast_median) + lag, :].shape,
              X_train_forecast_median.shape)

        testPredictPlot_transformer = np.empty_like(closedf)
        testPredictPlot_transformer[:, :] = np.nan
        testPredictPlot_transformer[len(X_train_forecast_median) + (lag * 2):len(closedf), :] = X_test_forecast_median
        print(testPredictPlot_transformer[len(X_train_forecast_median) + (lag * 2):len(closedf), :].shape,
              X_test_forecast_median.shape)

    if GMDH:
        if transformer:
            plotdf = pd.DataFrame({'date': close_stock['Date'],
                                   'original_close': close_stock['Close'],
                                   'train_predicted_close_arima': trainPredictPlot_arima.reshape(1, -1)[0].tolist(),
                                   'test_predicted_close_arima': testPredictPlot_arima.reshape(1, -1)[0].tolist(),
                                   'train_predicted_close': trainPredictPlot.reshape(1, -1)[0].tolist(),
                                   'test_predicted_close': testPredictPlot.reshape(1, -1)[0].tolist(),
                                   'train_predicted_close_gmdh': trainPredictPlot_gmdh.reshape(1, -1)[0].tolist(),
                                   'test_predicted_close_gmdh': testPredictPlot_gmdh.reshape(1, -1)[0].tolist(),
                                   'train_predicted_close_transformer': trainPredictPlot_transformer.reshape(1, -1)[0].tolist(),
                                   'test_predicted_close_transformer': testPredictPlot_transformer.reshape(1, -1)[0].tolist()})
        elif not transformer:
            plotdf = pd.DataFrame({'date': close_stock['Date'],
                                   'original_close': close_stock['Close'],
                                   'train_predicted_close_arima': trainPredictPlot_arima.reshape(1, -1)[0].tolist(),
                                   'test_predicted_close_arima': testPredictPlot_arima.reshape(1, -1)[0].tolist(),
                                   'train_predicted_close': trainPredictPlot.reshape(1, -1)[0].tolist(),
                                   'test_predicted_close': testPredictPlot.reshape(1, -1)[0].tolist(),
                                   'train_predicted_close_gmdh': trainPredictPlot_gmdh.reshape(1, -1)[0].tolist(),
                                   'test_predicted_close_gmdh': testPredictPlot_gmdh.reshape(1, -1)[0].tolist()})
    elif not GMDH:
        if transformer:
            plotdf = pd.DataFrame({'date': close_stock['Date'],
                                   'original_close': close_stock['Close'],
                                   'train_predicted_close_arima': trainPredictPlot_arima.reshape(1, -1)[0].tolist(),
                                   'test_predicted_close_arima': testPredictPlot_arima.reshape(1, -1)[0].tolist(),
                                   'train_predicted_close': trainPredictPlot.reshape(1, -1)[0].tolist(),
                                   'test_predicted_close': testPredictPlot.reshape(1, -1)[0].tolist(),
                                  'train_predicted_close_transformer': trainPredictPlot_transformer.reshape(1, -1)[0].tolist(),
                                   'test_predicted_close_transformer': testPredictPlot_transformer.reshape(1, -1)[0].tolist()})
        else:
            plotdf = pd.DataFrame({'date': close_stock['Date'],
                               'original_close': close_stock['Close'],
                               'train_predicted_close_arima': trainPredictPlot_arima.reshape(1, -1)[0].tolist(),
                               'test_predicted_close_arima': testPredictPlot_arima.reshape(1, -1)[0].tolist(),
                               'train_predicted_close': trainPredictPlot.reshape(1, -1)[0].tolist(),
                               'test_predicted_close': testPredictPlot.reshape(1, -1)[0].tolist()})
    fig, ax = plt.subplots()
    ax.plot(plotdf['date'], plotdf['original_close'], label='Оригинальная цена закрытия')
    ax.plot(plotdf['date'], plotdf['train_predicted_close_arima'], label='Предсказанная цена закрытия на тренировке SARIMA')
    ax.plot(plotdf['date'], plotdf['test_predicted_close_arima'], label='Предсказанная цена закрытия на тесте SARIMA')
    ax.plot(plotdf['date'], plotdf['train_predicted_close'], label='Предсказанная цена закрытия на тренировке')
    ax.plot(plotdf['date'], plotdf['test_predicted_close'], label='Предсказанная цена закрытия на тесте')
    if GMDH:
        ax.plot(plotdf['date'], plotdf['train_predicted_close_gmdh'], label='Предсказанная цена закрытия на тренировке GMDH')
        ax.plot(plotdf['date'], plotdf['test_predicted_close_gmdh'], label='Предсказанная цена закрытия на тесте GMDH')
    if transformer:
        ax.plot(plotdf['date'], plotdf['train_predicted_close_transformer'], label='Предсказанная цена закрытия на тренировке Transformer')
        ax.plot(plotdf['date'], plotdf['test_predicted_close_transformer'], label='Предсказанная цена закрытия на тесте Transformer')
    ax.legend()
    ax.set_title("Сравнение исходных и смоделированных цен")
    st.pyplot(fig)


    my_bar.progress(100, text='Done')


    if recursive_pred:
        lst_output_arima = make_prediction_recursive(test_data=test_data, method='SARIMA', model=arima_model,
                                                     scaler=scaler, pred_days=pred_days,
                                                     time_step_backward=time_step_backward)
        lst_output_lstm = make_prediction_recursive(test_data=test_data, method='LSTM', model=model,
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

        """
        x_input = test_data[len(test_data) - time_step_backward:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()


        lst_output = []
        n_steps = time_step_backward
        i = 0
        while (i < pred_days):

            if (len(temp_input) > time_step_backward):

                x_input = np.array(temp_input[1:])
                # print("{} day input {}".format(i,x_input))
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))

                yhat = model.predict(x_input, verbose=0)
                # print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                # print(temp_input)

                lst_output.extend(yhat.tolist())
                i = i + 1

            else:

                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())

                lst_output.extend(yhat.tolist())
                i = i + 1

        print("Output of predicted next steps: ", len(lst_output))
        """
        last_days = np.arange(1, time_step_backward + 1)
        day_pred = np.arange(time_step_backward + 1, time_step_backward + pred_days + 1)
        print(last_days)
        print(day_pred)

        temp_mat = np.empty((len(last_days) + pred_days, 1))
        temp_mat[:] = np.nan
        """
        last_original_days_value = temp_mat.copy()
        next_predicted_days_value = temp_mat.copy()
        last_original_days_value[0:time_step_backward] = closedf[len(closedf) - time_step_backward:].values
        next_predicted_days_value[time_step_backward:] = scaler.inverse_transform(np.array(lst_output))
        """
        last_original_days_value = temp_mat.copy()
        next_predicted_days_value_arima = temp_mat.copy()
        next_predicted_days_value_lstm = temp_mat.copy()
        if GMDH:
            next_predicted_days_value_gmdh = temp_mat.copy()
        if transformer:
            next_predicted_days_value_transformer = temp_mat.copy()

        last_original_days_value[0:time_step_backward] = \
            closedf[len(closedf) - time_step_backward:].values
        next_predicted_days_value_arima[time_step_backward:] = lst_output_arima
        next_predicted_days_value_lstm[time_step_backward:] = lst_output_lstm
        if GMDH:
            next_predicted_days_value_gmdh[time_step_backward:] = lst_output_gmdh
        if transformer:
            next_predicted_days_value_transformer[time_step_backward:] = lst_output_transformer

        """
        new_pred_plot = pd.DataFrame({
            'last_original_days_value': last_original_days_value.reshape(1, -1).tolist()[0],
            'next_predicted_days_value': next_predicted_days_value.reshape(1, -1).tolist()[0]
        })
        


        fig, ax = plt.subplots()
        ax.plot(new_pred_plot.index, new_pred_plot['last_original_days_value'], label=f"Последние {time_step_backward} шагов цены закратия")
        ax.plot(new_pred_plot.index, new_pred_plot['next_predicted_days_value'], label=f"Предсказанные следующие {pred_days} шагов цены закрытия")
        ax.legend()
        ax.set_title(f"Сравнения последних {time_step_backward} шагов и следующих {pred_days} шагов")
        st.pyplot(fig)
        """
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
                label=f"Последние {time_step_backward} шагов цены закратия")
        ax.plot(new_pred_plot.index, new_pred_plot['next_predicted_days_value_arima'],
                label=f"Предсказанные следующие {pred_days} шагов цены закрытия SARIMA")
        ax.plot(new_pred_plot.index, new_pred_plot['next_predicted_days_value_lstm'],
                label=f"Предсказанные следующие {pred_days} шагов цены закрытия LSTM")
        if GMDH:
            ax.plot(new_pred_plot.index, new_pred_plot['next_predicted_days_value_gmdh'],
                    label=f"Предсказанные следующие {pred_days} шагов цены закрытия GMDH")
        if transformer:
            ax.plot(new_pred_plot.index, new_pred_plot['next_predicted_days_value_transformer'],
                    label=f"Предсказанные следующие {pred_days} шагов цены закрытия Transformer")
        ax.legend()
        ax.set_title(f"Сравнения последних {time_step_backward} шагов и следующих {pred_days} шагов")
        ax.set_ylim(0, closedf['Close'].max() * 1.5)
        st.pyplot(fig)
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



import streamlit as st
st.set_page_config(layout="wide")
# Use the full page instead of a narrow central column
# st.set_page_config(layout="wide")

import seaborn as sns
from datetime import date
import time
from dateutil.relativedelta import relativedelta

from plotly import graph_objs as go
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
def app():
    global prediction_days


    def pg_bar():
        # rangebar = 0
        # Add a placeholder
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(100):
            latest_iteration.text(f'Iteration {i+1}')
            bar.progress(i + 1)
            time.sleep(0.1)

    # def normalize(dataset):
    #     scaled_data = scaler.fit_transform(dataset.values.reshape(-1,1))

    # df=pd.read_csv("/content/sample_data/hasil_sentiment_bulan.csv")
    # df=pd.read_csv("/content/drive/My Drive/Skripsi/hasil_sentiment_bulan.csv")

    st.title("Prediksi Sentiment")
    # df = ""
    # scaler = MinMaxScaler(feature_range=(0,1))


    # df = pd.read_csv('Riset Tre d/kategori_wisata_alam.csv')
    # try:
    #     upload_file = ""
    #     upload_file = st.file_uploader("upload data csv")
    #     df = pd.read_csv(upload_file)
    #     st.write(df.head())
    # except FileNotFoundError:
    #     st.error('File not found.')
    upload_file = st.file_uploader("upload data csv")
    if upload_file is None:
        st.info('File NOT found')
    else:
        name = st.text_input("Berikan nama kategori")
        st.write(name)
        df = pd.read_csv(upload_file)
        st.write(df.head())
        # positive_ = df['Positive']
        # negative_ = df['Negative']
        # neutral_ = df['Neutral']
        train_date =pd.to_datetime(df['Bulan'])
        # print(train_date)
        cols = list(df)[1:4]
        # print(cols)

        df_for_training = df[cols].astype(float)
        df_forplot = df_for_training.tail(5000)
        # df_forplot.plot.line()
        st.subheader('Plot Line dataset Sentiment')
        fig = go.Figure()
        st.line_chart(df_forplot)

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df['Positive'].values.reshape(-1,1))
        scaled_data1 = scaler.fit_transform(df['Negative'].values.reshape(-1,1))
        scaled_data0 = scaler.fit_transform(df['Neutral'].values.reshape(-1,1))

        st.subheader('Normalisasi Dataset')
        c5,c6,c7= st.columns((3,3,3))
        c5.subheader('Positive')
        c5.write(scaled_data)
        c6.subheader('Negative')
        c6.write(scaled_data1)
        c7.subheader('Neutral')
        c7.write(scaled_data0)

        # st.subheader('Normalisasi Dataset Neutral')
        # st.write(scaled_data0)

        # st.text(len(scaled_data))
        # prediction_days = 0
        st.subheader('Membuat Model x dan y dengan dataset')

        input_pred_test = st.slider('Masukkan jumlah bulan untuk model prediksi testing', min_value=1, max_value=30, value=5, step=1)
        # st.write(int_val)
        # input_pred_test = st.number_input("Masukkan jumlah hari untuk model prediksi testing")
        makemodel = st.text("Membuat model x dan y train . .")


            # '...pemodelan selesai'
                # makemodel.text("Selesai membuat model ..")

        # prediction_days = input_pred_test

        if input_pred_test == 0:
            st.text("masuukan kembali")
        else:
            try:
                prediction_days = 0
                prediction_days += input_pred_test
                st.subheader("Hasil x train")
                c8,c9,c10= st.columns((3,3,3))
                # st.text("data prediksi :"+prediction_days)
                # pg_bar() 
                x_train = []
                y_train = []

                for x in range(prediction_days, len(scaled_data)):
                    x_train.append(scaled_data[x - prediction_days:x, 0])
                    y_train.append(scaled_data[x, 0])
                # x_train
                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                c8.subheader("positive")
                c8.write(x_train[0])

                # Negative
                x_train1 = []
                y_train1 = []

                for x in range(prediction_days, len(scaled_data1)):
                    x_train1.append(scaled_data1[x - prediction_days:x, 0])
                    y_train1.append(scaled_data1[x, 0])
                # x_train1
                x_train1, y_train1 = np.array(x_train1), np.array(y_train1)
                x_train1 = np.reshape(x_train1, (x_train1.shape[0], x_train1.shape[1], 1))
                c9.subheader("negative")
                c9.write(x_train1[0])

                # Neutral
                x_train0 = []
                y_train0 = []

                for x in range(prediction_days, len(scaled_data0)):
                    x_train0.append(scaled_data0[x - prediction_days:x, 0])
                    y_train0.append(scaled_data0[x, 0])
                # x_train0
                x_train0, y_train0 = np.array(x_train0), np.array(y_train0)
                x_train0 = np.reshape(x_train0, (x_train0.shape[0], x_train0.shape[1], 1))
                c10.subheader("Neutral")
                c10.write(x_train0[0])
            except FileNotFoundError:
                st.error('File not found.')
            
            
        st.text("Model LSTM dibuat..")


        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()
        model.fit(x_train, y_train, batch_size=2, epochs=10)

        # Model Negatuve
        model1 = Sequential()
        model1.add(LSTM(50, return_sequences=True, input_shape=(x_train1.shape[1], 1)))
        model1.add(LSTM(50, return_sequences=False))
        model1.add(Dense(25))
        model1.add(Dense(1))

        # Compile model1
        model1.compile(optimizer='adam', loss='mean_squared_error')
        model1.summary()
        model1.fit(x_train1, y_train1, batch_size=2, epochs=10)

        # Model Neutral
        model0 = Sequential()
        model0.add(LSTM(50, return_sequences=True, input_shape=(x_train0.shape[1], 1)))
        model0.add(LSTM(50, return_sequences=False))
        model0.add(Dense(25))
        model0.add(Dense(1))

        # Compile model0
        model0.compile(optimizer='adam', loss='mean_squared_error')
        model0.summary()
        model0.fit(x_train0, y_train0, batch_size=2, epochs=10)

        # pg_bar()
        st.text("Model selesai dibuat..")

        n_future = 12
        input_pred_new = st.slider('Masukkan jumlah bulan yang ingin di prediksi', min_value=1, max_value=15, value=5, step=1)
        if input_pred_new >= 0:
            n_future = 0
            n_future += input_pred_new
        forecast_period_dates = pd.date_range(list(train_date)[+1],periods=n_future,freq='1m').tolist()
        forecast = model.predict(x_train[-n_future:])
        # Negative
        if input_pred_new >= 0:
            n_future = 0
            n_future += input_pred_new
        forecast_period_dates1 = pd.date_range(list(train_date)[+1],periods=n_future,freq='1m').tolist()
        forecast1 = model.predict(x_train1[-n_future:])
        # Neutral
        if input_pred_new >= 0:
            n_future = 0
            n_future += input_pred_new
        forecast_period_dates0 = pd.date_range(list(train_date)[+1],periods=n_future,freq='1m').tolist()
        forecast0 = model.predict(x_train0[-n_future:])
        st.subheader("Hasil Prediksi Positive")

        # st.write(forecast)
        # # Negative
        # st.subheader("Hasil Prediksi Negative")
        # st.write(forecast1)
        # # Neutral
        # st.subheader("Hasil Prediksi Neutral")
        # st.write(forecast0)


        forecast_copies = np.repeat(forecast,df_for_training.shape[1],axis=-1)
        y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]
        # Negative
        forecast_copies1 = np.repeat(forecast1,df_for_training.shape[1],axis=-1)
        y_pred_future1 = scaler.inverse_transform(forecast_copies1)[:,0]
        # Neutral
        forecast_copies0 = np.repeat(forecast0,df_for_training.shape[1],axis=-1)
        y_pred_future0 = scaler.inverse_transform(forecast_copies0)[:,0]

        # st.subheader("Mengembalikan nilai setelah di normalisasikan Positive")
        # st.write(y_pred_future)
        # st.subheader("Mengembalikan nilai setelah di normalisasikan Negative")
        # st.write(y_pred_future1)
        # st.subheader("Mengembalikan nilai setelah di normalisasikan Neutral")
        # st.write(y_pred_future0)



        # col1, col2 = st.columns(2)
        # col1.st.write(forecast, use_column_width=True)
        # col2.image(y_pred_future, use_column_width=True)


        forecast_dates = []
        for time_i in forecast_period_dates:
            forecast_dates.append(time_i.date())
        # forecast_dates1 = []
        # for time_i in forecast_period_dates1:
        #     forecast_dates1.append(time_i.date())
        # forecast_dates0 = []
        # for time_i in forecast_period_dates0:
        #     forecast_dates0.append(time_i.date())


        df_forecast = pd.DataFrame({'Bulan':np.array(forecast_dates),'Positive':y_pred_future})
        df_forecast['Bulan']=pd.to_datetime(df_forecast['Bulan'])
        # Negative
        df_forecast1 = pd.DataFrame({'Bulan':np.array(forecast_dates),'Negative':y_pred_future1})
        df_forecast1['Bulan']=pd.to_datetime(df_forecast1['Bulan'])
        # Neutral
        df_forecast0 = pd.DataFrame({'Bulan':np.array(forecast_dates),'Neutral':y_pred_future0})
        df_forecast0['Bulan']=pd.to_datetime(df_forecast0['Bulan'])

        ## Range selector
        # cols1,_ = st.beta_columns((1,2)) # To make it narrower
        # format = 'MMM DD, YYYY'  # format output
        # start_date = dt.date(year=2021,month=1,day=1)-relativedelta(years=2)  #  I need some range in the past
        # end_date = dt.datetime.now().date()-relativedelta(years=2)
        # max_days = end_date-start_date
        tanggal = '2020-9-1'
        # tanggal = cols1.slider('Pilih tanggal data original yang ingin ditampilkan', min_value=start_date, value=end_date ,max_value=end_date, format=format)
        # tanggal = st.text_input("Masukkan tanggal dan bulan untuk data original yang ingin ditampilkan ")
        # st.text("ex: 2019-9-1")



        # today = ''
        # tomorrow = today + datetime.timedelta(days=1)
        # start_date = st.date_input('Pilih tanggal data original yang ingin ditampilkan', today)
        original = df[['Bulan','Positive']]
        original['Bulan']=pd.to_datetime(original['Bulan'])
        original = original.loc[original['Bulan']>=tanggal]
        # Negative
        original1 = df[['Bulan','Negative']]
        original1['Bulan']=pd.to_datetime(original1['Bulan'])
        original1 = original1.loc[original1['Bulan']>=tanggal]
        # Neutral
        original0 = df[['Bulan','Neutral']]
        original0['Bulan']=pd.to_datetime(original0['Bulan'])
        original0 = original0.loc[original0['Bulan']>=tanggal]

        # st.subheader("Data original Positive")
        # st.write(original.tail())

        # st.subheader("Data original Negative")
        # st.write(original1)

        # st.subheader("Data original Neutral")
        # st.write(original0)

        # st.subheader("Data Forecast Positive")
        # st.write(df_forecast)

        # c1= st.columns((1))
        # st.subheader('Normalisasi Dataset')
        # c1.header("Data original Positive")
        # c1.write(original.tail())
        # c2.header("Data Forecast Positive")
        # c2.write(df_forecast)
        # st.subheader('Normalisasi Dataset')
        c11,c12,c13= st.columns((3,3,3))
        c11.subheader('Positive')
        c11.write(df_forecast)
        c12.subheader('Negative')
        c12.write(df_forecast1)
        c13.subheader('Neutral')
        c13.write(df_forecast0)
        # # Negative
        # st.subheader("Data Forecast Negative")
        # st.write(df_forecast1)
        # # Neutral
        # st.subheader("Data Forecast Neutral")
        # st.write(df_forecast0)
        positive_ee = df['Positive']
        negative_ee = df['Negative'].values
        neutral_ee = df['Neutral'].values
        from sklearn.metrics import mean_squared_error
        from numpy import sqrt
        c14,c15,c16= st.columns((3,3,3))
        c14.subheader("Hasil RMSE Positive")
        rmse = np.sqrt(np.mean(forecast-y_train)** 2)
        c14.write('RMSE : '+str(rmse))
        # Negative
        c15.subheader("Hasil RMSE Negative")
        rmse1 = np.sqrt(np.mean(forecast1 - y_train1) ** 2)
        c15.write('RMSE1 : '+str(rmse1))
        # Neutral
        c16.subheader("Hasil RMSE Neutral")
        rmse0 = np.sqrt(np.mean(forecast0 - y_train0) ** 2)
        c16.write('RMSE0 : '+str(rmse0))



        frames = [original, df_forecast]
        result = pd.concat(frames)
        # Negative
        frames1 = [original1, df_forecast1]
        result1 = pd.concat(frames1)
        # Neutral
        frames0 = [original0, df_forecast0]
        result0 = pd.concat(frames0)


        # st.subheader("Grafik Hasil Prediksi Positive")
        # df_forecast = df_forecast.rename(columns={'Bulan':'index'}).set_index('index')
        # df_forecast1 = df_forecast1.rename(columns={'Bulan':'index'}).set_index('index')
        # df_forecast0 = df_forecast0.rename(columns={'Bulan':'index'}).set_index('index')

        # st.line_chart(df_forecast)

        # st.subheader(f"Grafik Gabungan Data Original dari {tanggal} dan Prediksi ke {n_future} ")



        # jumlah = 0
        # for x in df_forecast:
        #     jumlah += x
        # rata_2 = jumlah / len(df_forecast)
        # st.line_chart(result)
        # origin_pos = sns.lineplot(original['Bulan'],original['Positive'])
        # forecast_pos = sns.lineplot(df_forecast['Bulan'],df_forecast['Positive'])
        # origin_neg = sns.lineplot(original1['Bulan'],original1['Negative'])
        # forecast_neg = sns.lineplot(df_forecast1['Bulan'],df_forecast1['Negative'])
        # origin_net = sns.lineplot(original0['Bulan'],original0['Neutral'])
        # forecast_net = sns.lineplot(df_forecast0['Bulan'],df_forecast0['Neutral'])
        result = result.rename(columns={'Bulan':'index'}).set_index('index')

        c17,c18,c19 = st.columns((3,3,3))
        c17.header("Grafik Hasil Prediksi Positive")
        result = result.rename(columns={'Positive':name})
        c17.line_chart(result)
        st.download_button(label='Download CSV Positive',file_name='trend positive.csv',data=result.to_csv(),mime='text/csv')

        c18.header("Grafik Hasil Prediksi Negative")
        result1 = result1.rename(columns={'Bulan':'index'}).set_index('index')
        result1 = result1.rename(columns={'Negative':name})
        c18.line_chart(result1)
        st.download_button(label='Download CSV Negative',file_name='trend Negative.csv',data=result1.to_csv(),mime='text/csv')


        c19.header("Grafik Hasil Prediksi Neutral")
        result0 = result0.rename(columns={'Bulan':'index'}).set_index('index')
        result0 = result0.rename(columns={'Neutral':name})
        c19.line_chart(result0)
        st.download_button(label='Download CSV Netral',file_name='trend Netral.csv',data=result0.to_csv(),mime='text/csv')


        
    #c.Output = "/content/drive/My Drive/Skripsi/wisata sumenep.csv"


    # positive_ = df['Positive']
    # negative_ = df['Negative']
    # neutral_ = df['Neutral']

    # st.plotly_chart(df_forplot)
    # df.head()
    # df_for_training


    

    # scaled_data1 = df['Negative'].values.reshape(-1,1)

    # normalize(df['Positive'])



    # scaled_data1 = df['Negative'].values.reshape(-1,1)

    # scaled_data0 = df['Neutral'].values.reshape(-1,1)


    # print(scaled_data)
    # st.subheader('Normalisasi Dataset Positive')
    # st.write(scaled_data)

    # st.subheader('Normalisasi Dataset Negative')
    # st.write(scaled_data1)
   

    # # df_forecast1 = df_forecast1.rename(columns={'Bulan':'index'}).set_index('index')
    # c18.header("Grafik Hasil Prediksi Negative")
    # c18.line_chart(df_forecast1)
    # c19.header("Grafik Hasil Prediksi Neutral")
    # # df_forecast0 = df_forecast0.rename(columns={'Bulan':'index'}).set_index('index') 
    # c19.line_chart(df_forecast0)
    # c4.header(f"Grafik Gabungan Data Original dari {tanggal} dan Prediksi ke {n_future} ")
    # c4.line_chart(result)
    # # Negative
    # st.subheader("Grafik Hasil Prediksi Negative")
    # df_forecast1 = df_forecast1.rename(columns={'Date':'index'}).set_index('index')
    # st.line_chart(df_forecast1)

    # st.subheader(f"Grafik Gabungan Data Original dari {tanggal} dan Prediksi ke {n_future} ")
    # result1 = result1.rename(columns={'Date':'index'}).set_index('index')
    # st.line_chart(result1)
    # # Neutral
    # st.subheader("Grafik Hasil Prediksi Neutral")
    # df_forecast0 = df_forecast0.rename(columns={'Date':'index'}).set_index('index')
    # st.line_chart(df_forecast0)

    # st.subheader(f"Grafik Gabungan Data Original dari {tanggal} dan Prediksi ke {n_future} ")
    # result0 = result0.rename(columns={'Date':'index'}).set_index('index')
    # st.line_chart(result0)



    # st.line_chart(original[[‘Closed’]])
    # st.line_chart(df_forecast)

    # print('\nRoot mean square (RMSE):' + str(rmse))

    # fig1 = plot_plotly()
    # print(original)
    # print(df_forecast)
    # forecast_copies







    # st.text("jumlah prediksi"+prediction_days)









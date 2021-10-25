import streamlit as st
# st.set_page_config(layout="wide")
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd
from functools import reduce

def app():
    st.title('Trend Visualization')
    all_df = []

    multiple_files = st.file_uploader('CSV',type="csv", accept_multiple_files=True)
    # df1 = st.file_uploader('CSV',type="csv", accept_multiple_files=True)
    # df2 = st.file_uploader('CSV',type="csv", accept_multiple_files=True)
    for file in multiple_files:
        dataframe = pd.read_csv(file, sep=',')
        all_df.append(dataframe)
        # df3 = pd.merge(all_df)
        
        # print(dataframe)
        name = ""
        # for f in all_df:
        #     indexx = all_df[x]
        #     x+=1
        #     merged_ = pd.merge(indexx)
        # st.write(merged_)
        
        # print(dataframe.columns)
        # st.write(dataframe[['Action']])
        # for df in dataframe:
        #     gambar = sns.lineplot(dataframe.index,dataframe[df])
        #     # st.pyplot(gambar.figure) 
        #     # name = df
        #     st.write(df)
        # merged_ = pd.merge(all_df)
        # print(all_df)
        
            # merged_ = pd.merge(merged_[0],f,how="right")
                # st.write(df)
        
        # st.write(dataframe.iloc[1])
        # c17.write(dataframe.head())
        # c18.line_chart(dataframe)
        # st.write(merged_.head())
        # st.line_chart(merged_)
        # st.download_button(label='Download CSV Netral',file_name='trend Netral.csv',data=merged_.to_csv(),mime='text/csv')
    c17,c18 = st.columns((2,5))
        
        # x = 0
        # for f in all_df:
        #     indexx = all_df[x]
        #     x+=1
        #     merged_ = pd.merge(indexx)
        # st.write(merged_)
        
        # c17.write(dataframe)
    try:
        df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['index'],
                                        how='outer'), all_df)
        df_merged = df_merged.rename(columns={'index':'index'}).set_index('index')
        
        c17.write(df_merged)
        # result0 = result0.rename(columns={'Bulan':'index'}).set_index('index')

        c18.line_chart(df_merged)

            
        file.seek(0)
    except:
    # Prevent the error from propagating into your Streamlit app.
        pass
        
    
    

    

    #     df3 = pd.merge(df,how="right")
    #     st.write(df3.head())
        # st.download_button(label='Download CSV',file_name='trend '+name+'.csv',data=dataframe.to_csv(),mime='text/csv')

            # st.write(df.values)
        # st.subheader(f"")
    # st.subheader(f"Grafik Gabungan Data Original dari {tanggal} dan Prediksi ke {n_future} ")

        # print(dataframe.iloc[df])
        # st.write(dataframe)
        # st.line_chart(dataframe)
        # st.pyplot(gambar.figure) 
        
        # result0 = dataframe.rename(columns={'Bulan':'index'}).set_index('index')
        # gambar = sns.lineplot(result0.Bulan,result0.loc[1])
        # result0 = result0.rename(columns={'Bulan':'index'}).set_index('index')
        # st.pyplot(gambar.figure) 
    # result0 = dataframe.rename(columns={'Bulan':'index'}).set_index('index')
    # list(dataframe)[1]
    # gambar = sns.lineplot(result0.index,result0.loc[1])
    # result0 = result0.rename(columns={'Bulan':'index'}).set_index('index')
    # st.pyplot(gambar.figure)    

    #     # train_date =pd.to_datetime(dataframe['Bulan'])
    #     # print(train_date)
    # cols = list(dataframe)[1:]
    #     # print(cols)

    # df_for_training = dataframe[cols].astype(float)
    # df_forplot = df_for_training.tail(5000)
    # # df_forplot.plot.line()
    # st.subheader('Plot Line dataset Sentiment')
    # fig = go.Figure()
    
    # df_for_training = df_for_training.rename(columns={'Bulan':'index'}).set_index('index')
    # st.line_chart(df_forplot)    
        
    # train_date =pd.to_datetime(dataframe['Bulan'])
    # # print(train_date)
    # cols = list(dataframe)[1:4]
    # # print(cols)

    # df_for_training = dataframe[cols].astype(float)
    # df_forplot = df_for_training.tail(5000)
    # # df_forplot.plot.line()
    # st.subheader('Plot Line dataset Sentiment')
    # fig = go.Figure()
    # # df_for_training = df_for_training.rename(columns={'Bulan':'index'}).set_index('index')
    # st.line_chart(df_forplot)
    # import pandas as pd
    # @st.cache
    # def do_a_thing_with_uploaded_file(value):
    #     pass
    # # data = pd.read_csv(r'C:\Users\Afan\Documents\Skripsi\Skripsi_september\genre action 20-21\trend_positive.csv')
    # # st.write(data.head())

    # # result = []
    # # data["Action"] = ['1','2','3','4','5','6','7','1','2','3','4','5','6','7','1','2','4','5']
    # # data_saved = data
    # # do_a_thing_with_uploaded_file(data_saved)
    # st.write(data_saved.head())



    # # Load iris dataset
    # iris = datasets.load_iris()
    # X = iris.data
    # Y = iris.target

    # # Model building
    # st.header('Model performance')
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.2, random_state=42)
    # clf = RandomForestClassifier()
    # clf.fit(X_train, Y_train)
    # score = clf.score(X_test, Y_test)
    # st.write('Accuracy:')
    # st.write(score)

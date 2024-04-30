import streamlit as st
import pandas as pd
import datetime

st.set_page_config(
    page_title='View History',
    page_icon=':)',
    layout='wide'
)

st.title('History of Prediction')

def display_View_History_of_Prediction():
    #path = 'Assets/Datasets/ViewHistory.csv'
    #path = '.\Assets\Datasets\ViewHistory.csv'
    #df = pd.read_csv(path)
    return df
st.markdown("<h1 style='text-align:center;'>🕰️ Chronicles of Past Forecasts ⏳</h1>", unsafe_allow_html=True)

if __name__ == '__main__':
   df = display_View_History_of_Prediction()
   st.dataframe(df)

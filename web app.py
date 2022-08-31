import pandas_datareader as data  # for data Scraping
import datetime
import streamlit as st
from keras.model import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

start = "2010-01-01"  # YYYY-MM-DD
# end = "2020-12-31"
end = datetime.datetime.now()

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'TTM')

df = data.DataReader(user_input, 'yahoo', start, end)

# Describing Data
st.subheader('Data from 2010 till now')
st.write(df.describe())

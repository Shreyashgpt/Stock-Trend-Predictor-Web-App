import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as data  # for data Scraping
import datetime
import streamlit as st
from keras.models import load_model


start = "2010-01-01"  # YYYY-MM-DD
# end = "2020-12-31"
end = datetime.datetime.now()

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'TTM')

df = data.DataReader(user_input, 'yahoo', start, end)

# Describing Data
st.subheader('Data from 2010 till now')
st.write(df.describe())

# Visulations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 & 200 Moving Average')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100, label='100 Moving Average')
plt.plot(ma200, label='200 Moving Average')
plt.legend()
st.pyplot(fig)

# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0: int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

# Scaling Training Data

scalar = MinMaxScaler(feature_range=(0, 1))

data_training_array = scalar.fit_transform(data_training)


# Loading model
model = load_model('keras_model.h5')

# Testing Part
# for prediction of 2231st value we need data for past 100 days that is in training data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing])
final_df = final_df.reset_index(drop=True)

# Scaling Test Data
input_data = scalar.fit_transform(final_df)

# Splitting Test Data into X and Y
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making Prediction
y_predicted = model.predict(x_test)

# Scaling Predicted Values
scale_factor = 1 / scalar.scale_[0]

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plotting Predicted Values
st.subheader('Prediction Vs Original Price')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)

# Prediction Into Future
st.subheader('Trend Prediction')
n = int(st.number_input('Enter number of days to predict',
        min_value=1, value=15, step=1))

# df = data.DataReader('TTM', 'yahoo', start, end)
# df = df.reset_index()
# df = df.drop(['Date', 'Adj Close', 'High', 'Low', 'Volume', 'Open'], axis=1)

# # Scaling
# input_future = np.array(scalar.fit_transform(df))

input_data = input_data[-100:]  # for last 100 values only

x = []
y = []
x.append(input_data)
x = np.array(x)

for i in range(100, 100+n):
    op = model.predict(x)
    y.append(op)
    x = np.delete(x, 0, axis=1)
    x = np.append(x, op)
    x = x.reshape(1, 100, 1)

y = np.array(y)
y = y.reshape(n,)

y = y / scalar.scale_[0]

fig3 = plt.figure(figsize=(12, 6))
plt.plot(y)
plt.xlabel('Time')
plt.ylabel('Prices')
# plt.show
st.pyplot(fig3)

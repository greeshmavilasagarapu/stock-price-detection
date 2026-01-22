import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

st.title('AI Stock Trend Predictor')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start='2010-01-01', end='2023-12-31')

# Data Visualizations
st.subheader('Data Summary (2010 - 2023)')
st.write(df.describe())

st.subheader('Closing Price vs Time')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Price vs 100 & 200 Day Moving Averages')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig2 = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'b', label='Close')
plt.plot(ma100, 'r', label='100MA')
plt.plot(ma200, 'g', label='200MA')
plt.legend()
st.pyplot(fig2)

# Load Model
model = load_model('stock_model.h5')

# Prepare Testing Data
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Plot
st.subheader('Predictions vs Original')
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)
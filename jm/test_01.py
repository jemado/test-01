

# new comment JM (9.6.18)

# Marc hol mol Bier -> MAAAAARRRRC!!!!
# Whhhaaattt??


#%%
#-----------------------------------------------------------------------------------
# load libraries and co
#-----------------------------------------------------------------------------------
#pylint: disable=C0103, E1127
from datetime import datetime
import pandas as pd
import numpy as np
from binance.client import Client
#from external_functions import date_to_milliseconds, interval_to_milliseconds
<<<<<<< HEAD
# test comment
# Test Change Marc
# lkj lkaj sdf
=======
# test comment: funktioniert push auf github?
>>>>>>> c0ac1edd3890a36202add822547cd2019c484453
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot
import cufflinks as cf

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import dask

# required for plotting
init_notebook_mode(connected=True)
cf.go_offline()

#-----------------------------------------------------------------------------------
# Testzeile Marc
#%%
#-----------------------------------------------------------------------------------
# connect to Binance and get data
#-----------------------------------------------------------------------------------
api_key = 'mXSRegcVk837qVdQTapOeIYzw4RSqlMiICHtVq31rxKWdCMFoTB8Mtf5UtfWHLhd'
api_secret = 'UcWtExAC4Xw6avdaB8DK5C4kzaa3C4vgPvtezpfmeFrvGeTAurfPM5iAqKUWCkgU'

client = Client(api_key, api_secret)

# get market depth
depth = client.get_order_book(symbol='BNBBTC')

# get all symbol prices
prices = client.get_all_tickers()

# Get Recent Trades
trades = client.get_recent_trades(symbol='BNBBTC')

# Get Historical Trades
trades = client.get_historical_trades(symbol='BNBBTC')

# fetch 30 minute klines for the last month of 2017
interval = Client.KLINE_INTERVAL_30MINUTE
klines_ETH = client.get_historical_klines('ETHUSDT', interval,
                                      '26 Apr, 2017', '21 May, 2018')
klines_BTC = client.get_historical_klines('BTCUSDT', interval,
                                      '26 Apr, 2017', '21 May, 2018')
# info about klines format:

# https://python-binance.readthedocs.io/en/latest/binance.html#binance.client.Client.get_klines
#-----------------------------------------------------------------------------------

#%%
#-----------------------------------------------------------------------------------
# convert API data to pandas dataframe
#-----------------------------------------------------------------------------------
ts_ETH = pd.DataFrame(klines_ETH)
ts_ETH.columns = ['open_time', 'open', 'high', 'low', 'close',
              'volume', 'close_time', 'quote_asset_vol', 'num_trades',
              'taker_buy_base_asset_vol', 'taker_buy_quote_asset_vol', 'ignore']
ts_ETH['open_time'] = pd.to_datetime(ts_ETH['open_time'], unit='ms')
ts_ETH = ts_ETH.set_index('open_time')

ts_BTC = pd.DataFrame(klines_BTC)
ts_BTC.columns = ['open_time', 'open', 'high', 'low', 'close',
              'volume', 'close_time', 'quote_asset_vol', 'num_trades',
              'taker_buy_base_asset_vol', 'taker_buy_quote_asset_vol', 'ignore']
ts_BTC['open_time'] = pd.to_datetime(ts_BTC['open_time'], unit='ms')
ts_BTC = ts_BTC.set_index('open_time')

print(ts_BTC.tail())
#ts.info()


#-----------------------------------------------------------------------------------

#%%
#-----------------------------------------------------------------------------------
# plot data
#-----------------------------------------------------------------------------------
trace1 = go.Scatter(x = ts_ETH.index, y = ts_ETH.open)
data = [trace1]
layout = go.Layout()
fig1 = go.Figure(data = data, layout = layout)
plot(fig1, filename = 'plots/fig1.html')
#-----------------------------------------------------------------------------------

#%%
#-----------------------------------------------------------------------------------
# data manipulation
#-----------------------------------------------------------------------------------
# combine time series
ts = pd.concat([ts_BTC['close'], ts_ETH['close']], axis=1)
ts.columns = ['BTC', 'ETH']

# split into train and test sets
# train set
start_date = '2017-08-17 04:00:00'
end_date = '2017-10-17 04:00:00'
mask = (ts.index >= start_date) & (ts.index <= end_date)
train_data = ts.loc[mask]

# test set
start_date = '2018-03-26 00:00:00'
end_date = '2018-04-26 00:00:00'
mask = (ts.index >= start_date) & (ts.index <= end_date)
test_data = ts.loc[mask]
#-----------------------------------------------------------------------------------

#%%
#-----------------------------------------------------------------------------------
# Prediction with TensorFlow
#-----------------------------------------------------------------------------------
# combine time series
ts = pd.concat([ts_BTC['close'], ts_ETH['close']], axis=1)
ts.columns = ['BTC', 'ETH']

# clean data
# drop NAN entries
ts = ts.dropna()

# split into train and test sets
# train set
start_date = '2017-08-17 04:00:00'
end_date = '2018-02-17 04:00:00'
mask = (ts.index >= start_date) & (ts.index <= end_date)
data_train = ts.loc[mask]

# test set
start_date = '2018-03-26 00:00:00'
end_date = '2018-05-20 00:00:00'
mask = (ts.index >= start_date) & (ts.index <= end_date)
data_test = ts.loc[mask]

trace1 = go.Scatter(x = data_train.index, y = data_train['BTC'], name='train data')
trace2 = go.Scatter(x = data_test.index, y = data_test['BTC'], name='test data')
data = [trace1, trace2]
layout = go.Layout()
fig1 = go.Figure(data = data, layout = layout)
plot(fig1, filename = 'plots/fig1.html')

#%%
# check if there is a nan entry
#print(np.any(np.isnan(A)))

# Scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train[['BTC', 'ETH']].values.astype(float))
data_train_trans = scaler.transform(data_train[['BTC', 'ETH']].values.astype(float))
data_test_trans = scaler.transform(data_test[['BTC', 'ETH']].values.astype(float))

# Build X and y
X_train = data_train_trans[:, 1:]
y_train = data_train_trans[:, 0]
X_test = data_test_trans[:, 1:]
y_test = data_test_trans[:, 0]

# Number of stocks in training data
n_stocks = X_train.shape[1]

# Neurons
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128

# Session
net = tf.InteractiveSession()

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

#%%

# Initializers
sigma = 1
weight_initializer = tf.contrib.layers.variance_scaling_initializer(
    factor=2.0,
    mode='FAN_IN',
    uniform=False,
    seed=None,
    dtype=tf.float32
)
#bias_initializer = tf.zeros_initializer()

#print(data_train_trans[:, 1:])
#print(data_train_trans[:, 0])
#print(data_train_trans)
#print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

#data_test = scaler.transform(data_train['close'].values)
# data_test = scaler.transform(data_test)

# trace1 = go.Scatter(x = data_train.index, y = data_train, name='train data')
# trace2 = go.Scatter(x = data_test.index, y = data_test, name='test data')
# data = [trace1, trace2]
# layout = go.Layout()
# fig1 = go.Figure(data = data, layout = layout)
# plot(fig1, filename = 'jm/plots/fig2.html')

#-----------------------------------------------------------------------------------

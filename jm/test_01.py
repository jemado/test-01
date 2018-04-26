
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

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot
import cufflinks as cf

# required for plotting
init_notebook_mode(connected=True)
cf.go_offline()
#-----------------------------------------------------------------------------------

#%%
#-----------------------------------------------------------------------------------
# connect to Binance and get data
#-----------------------------------------------------------------------------------
api_key = 'mXSRegcVk837qVdQTapOeIYzw4RSqlMiICHtVq31rxKWdCMFoTB8Mtf5UtfWHLhd'
api_secret = 'UcWtExAC4Xw6avdaB8DK5C4kzaa3C4vgPvtezpfmeFrvGeTAurfPM5iAqKUWCkgU'

client = Client(api_key, api_secret)

# get market depth
depth = client.get_order_book(symbol='BNBBTC')

# Get Recent Trades
trades = client.get_recent_trades(symbol='BNBBTC')

# Get Historical Trades
trades = client.get_historical_trades(symbol='BNBBTC')

# fetch 30 minute klines for the last month of 2017
interval = Client.KLINE_INTERVAL_30MINUTE
klines = client.get_historical_klines('ETHUSDT', interval,
                                      '26 Apr, 2017', '26 Apr, 2018')
# info about klines format:
# https://python-binance.readthedocs.io/en/latest/binance.html#binance.client.Client.get_klines
#-----------------------------------------------------------------------------------

#%%
#-----------------------------------------------------------------------------------
# convert API data to pandas dataframe
#-----------------------------------------------------------------------------------
ts = pd.DataFrame(klines)
ts.columns = ['open_time', 'open', 'high', 'low', 'close',
              'volume', 'close_time', 'quote_asset_vol', 'num_trades',
              'taker_buy_base_asset_vol', 'taker_buy_quote_asset_vol', 'ignore']
ts['open_time'] = pd.to_datetime(ts['open_time'], unit='ms')
ts = ts.set_index('open_time')
print(ts.tail())
#ts.info()
#-----------------------------------------------------------------------------------

#%%
#-----------------------------------------------------------------------------------
# plot data
#-----------------------------------------------------------------------------------
trace1 = go.Scatter(x = ts.index, y = ts.open)
data = [trace1]
layout = go.Layout()
fig1 = go.Figure(data = data, layout = layout)
plot(fig1, filename = 'jm/plots/fig1.html')
#-----------------------------------------------------------------------------------

#%%
#-----------------------------------------------------------------------------------
# data manipulation
#-----------------------------------------------------------------------------------
# split into train and test sets
# train set
start_date = '2017-08-17 04:00:00'
end_date = '2017-10-17 04:00:00'
mask = (ts.index >= start_date) & (ts.index <= end_date)
train_data = ts['close'].loc[mask]

# test set
start_date = '2018-03-26 00:00:00'
end_date = '2018-04-26 00:00:00'
mask = (ts.index >= start_date) & (ts.index <= end_date)
test_data = ts['close'].loc[mask]

print(test_data)
#-----------------------------------------------------------------------------------
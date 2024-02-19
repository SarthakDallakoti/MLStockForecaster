import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def fetch_data(ticker_symbol, start_date, end_date):
    # Fetch historical data from Yahoo Finance
    ticker_data = yf.Ticker(ticker_symbol)
    ticker_df = ticker_data.history(start=start_date, end=end_date)
    return ticker_df

def add_moving_averages(ticker_df):
    # Calculate short-term and long-term Simple Moving Averages (SMA)
    ticker_df['SMA_short'] = ticker_df['Close'].rolling(window=10).mean()
    ticker_df['SMA_long'] = ticker_df['Close'].rolling(window=30).mean()
    return ticker_df

def identify_trend(ticker_df):
    # Determine the current trend based on SMA relationships
    if ticker_df['SMA_short'].iloc[-1] > ticker_df['SMA_long'].iloc[-1]:
        return 'Uptrend'
    elif ticker_df['SMA_short'].iloc[-1] < ticker_df['SMA_long'].iloc[-1]:
        return 'Downtrend'
    else:
        return 'Sideways'

def find_support_resistance(ticker_df, order=5):
    # Identify indices of local minima and maxima as support and resistance levels
    minima_indices = argrelextrema(ticker_df['Close'].values, np.less_equal, order=order)[0]
    maxima_indices = argrelextrema(ticker_df['Close'].values, np.greater_equal, order=order)[0]

    # Mark support and resistance levels
    ticker_df['Support'] = np.nan
    ticker_df['Resistance'] = np.nan
    ticker_df.iloc[minima_indices, ticker_df.columns.get_loc('Support')] = ticker_df['Close'][minima_indices]
    ticker_df.iloc[maxima_indices, ticker_df.columns.get_loc('Resistance')] = ticker_df['Close'][maxima_indices]
    return ticker_df

# Example usage
ticker_symbol = '^GSPC'  # S&P 500 index as an example
start_date = '2022-08-15'
end_date = '2022-10-14'

ticker_df = fetch_data(ticker_symbol, start_date, end_date)
ticker_df = add_moving_averages(ticker_df)
current_trend = identify_trend(ticker_df)
ticker_df = find_support_resistance(ticker_df, order=5)

print(f"Current Trend: {current_trend}")
print(ticker_df[['Close', 'SMA_short', 'SMA_long', 'Support', 'Resistance']].tail(20))

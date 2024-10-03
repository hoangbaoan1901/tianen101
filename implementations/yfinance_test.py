import yfinance as yf
import pandas as pd


data = yf.download("NVDA", start="2010-01-01", end="2024-01-01")
data = data.asfreq('D')  # Fill missing dates with NaN
data.reset_index(inplace=True)
data.rename(columns={'index': 'Date'}, inplace=True)
data.to_csv('/home/hoangbaoan1901/Development/predictive-analysis/Homeworks/week-3-btl/datasets/stocks.csv', index=False)
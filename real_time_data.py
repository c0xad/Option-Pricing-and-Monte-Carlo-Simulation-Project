# real_time_data.py

import yfinance as yf
import pandas as pd
import datetime

def fetch_real_time_data(ticker, start_date=None, end_date=None, interval='1d'):
    """
    Fetch real-time market data for a given ticker symbol.

    Parameters:
        ticker (str): The ticker symbol of the asset.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Data interval (e.g., '1d', '1h', '1m').

    Returns:
        pd.DataFrame: DataFrame containing the market data.
    """
    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.datetime.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')

    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"No data fetched for ticker {ticker}. Please check the ticker symbol and try again.")
    data.reset_index(inplace=True)
    return data

# class for loading, processing, and deviding to train and test examples
# space feature prepration
import yfinance as yf
import ta
import numpy as np
import pandas as pd


def enhance_dataframe_with_technical_indicators(df):
    """
    Enhances a given DataFrame with additional columns for normalized close prices,
    normalized returns for different periods, MACD, and RSI.

    Parameters:
    - df: DataFrame containing at least a 'close' column.

    Returns:
    - DataFrame with added columns.
    """
    # Ensure 'close' column exists
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    # Normalize close prices
    df['normalized_close'] = (df['Close'] - df['Close'].mean()) / df['Close'].std()

    # Calculate normalized returns for different periods
    for period in [30, 60, 90, 252]:  # Example periods
        df[f'return_{period}d'] = df['Close'].pct_change(periods=period) * np.sqrt(252 / period)

    # MACD
    df['macd'] = ta.trend.MACD(df['Close']).macd()

    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=30).rsi()

    df['daily_return'] = df['Close'].pct_change()  # Daily returns
    for window in [30, 60]:  # Example window sizes
        df[f'volatility_{window}d'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)  # Annualized Volatility

    return df.dropna()


def load_and_process_data(tickers, features, combine=False):
    """
    Load historical data, select specified features, enhance with technical indicators,
    optionally combine into a single DataFrame.
    """
    combined_data = []

    for pair, ticker in tickers.items():
        data = yf.Ticker(ticker).history(period="max", start="2005-01-01", end="2019-12-31")

        if 'Close' in data.columns:
            data = data[features]
            data = enhance_dataframe_with_technical_indicators(data)
            data['Currency_Pair'] = pair  # Add currency pair identifier

            # Append the DataFrame to the list for later combination
            combined_data.append(data)
        else:
            print(f"'Close' column not found or not selected for {pair}. Skipping.")

    if combine:
        # Combine all DataFrames into one; they should already have date as the index
        combined_data = pd.concat(combined_data)
        return combined_data
    else:
        # If not combining, you might return a dictionary of DataFrames, adjusting as needed
        return {pair: df for pair, df in zip(tickers.keys(), combined_data)}


def split_data(data, split_date="2016-01-01"):
    """
    Splits the data into training and testing sets based on a split date.
    """
    train_data = {}
    test_data = {}
    for pair, df in data.items():
        train_data[pair] = df.loc[:split_date]
        test_data[pair] = df.loc[split_date:]
    return train_data, test_data


def load_data():
    tickers = {
        "AUD/USD": "FXA",
        "GBP/USD": "GBPUSD=X",
        "USD/CAD": "CAD=X",
        "US Dollar Index": "UUP",
        "EUR/USD": "FXE",
        "JPY/USD": "FXY",
        "MXN/USD": "MXNUSD=X",
        "Nikkei 225": "EWJ",
        "CHF/USD": "CHFUSD=X",
    }
    # Initialize a dictionary to store DataFrame for each ticker
    hist_data = {}
    for pair, ticker in tickers.items():
        # Fetch historical data from Yahoo Finance
        data = yf.Ticker(ticker).history(period="1d", start="2011-01-01", end="2019-12-31")

        # Check if 'Close' column is in the data and proceed with enhancement
        if 'Close' in data.columns:
            # Enhance the DataFrame with technical indicators
            enhanced_data = enhance_dataframe_with_technical_indicators(data)
            # Store the enhanced DataFrame in the dictionary
            hist_data[pair] = enhanced_data
        else:
            # Handle cases where 'Close' column is missing by storing an empty DataFrame or logging an error
            print(f"'Close' column not found for {pair}. Skipping.")
            hist_data[pair] = pd.DataFrame()


    # Assuming df is your DataFrame with historical data including 'Close' and 'macd' columns
    df = enhanced_data[['Close', 'macd']]
    return  df.dropna()

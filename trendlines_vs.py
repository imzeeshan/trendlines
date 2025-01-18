import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from scipy.stats import linregress
import mysql.connector


def detect_trendlines(df, price_col='close', order=5):
    """
    Detect trendlines on a DataFrame.

    Parameters:
    - df: DataFrame with price data
    - price_col: Column containing price data to analyze
    - order: Sensitivity for finding extrema (higher = less sensitive)

    Returns:
    - DataFrame with extrema points and slopes of trendlines
    """
    # Find local maxima and minima
    maxima_idx = argrelextrema(df[price_col].values, np.greater, order=order)[0]
    minima_idx = argrelextrema(df[price_col].values, np.less, order=order)[0]

    # Extract extrema points
    maxima = df.iloc[maxima_idx]
    minima = df.iloc[minima_idx]

    # Fit trendlines
    max_slope, max_intercept = linregress(maxima.index, maxima[price_col])[:2]
    min_slope, min_intercept = linregress(minima.index, minima[price_col])[:2]

    # Compute trendline values
    df['max_trendline'] = max_slope * df.index + max_intercept
    df['min_trendline'] = min_slope * df.index + min_intercept

    return df, maxima, minima, max_slope, min_slope


def connect_local_maxima(df, price_col='close', order=5):
    """
    Detect and connect local maxima with lines on a DataFrame.

    Parameters:
    - df: DataFrame with price data
    - price_col: Column containing price data to analyze
    - order: Sensitivity for finding maxima (higher = less sensitive)

    Returns:
    - DataFrame with maxima points
    """
    # Find local maxima
    maxima_idx = argrelextrema(df[price_col].values, np.greater, order=order)[0]

    # Extract maxima points
    maxima = df.iloc[maxima_idx]

    # Plot the result
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[price_col], label='Close Price', color='blue', alpha=0.7)
    plt.scatter(maxima.index, maxima[price_col], color='orange', label='Local Maxima')

    # Connect maxima with lines
    plt.plot(maxima.index, maxima[price_col], color='red', linestyle='-', label='Trendline (Maxima)')

    # Customize the plot
    plt.title('Connecting Local Maxima')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return maxima

connection = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="zeeshan",
    password="zeeshan1983",
    database="lucrumbot"
)


df = pd.read_sql_query("select symbol,date, open,high,low ,close from futures_price where date >= '2025-01-16 00:00:00' and symbol='NQ'", con=connection)

maxima = connect_local_maxima(df, price_col='close', order=2)

# Print maxima
print("Local Maxima:")
print(maxima)
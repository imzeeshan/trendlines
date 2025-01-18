import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings
from typing import List, Tuple, Optional


def get_stock_data(
        symbol: str,
        days_back: int = 400,
        end_date: Optional[dt.date] = None
) -> Optional[pd.DataFrame]:
    """
    Download stock data from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol
        days_back: Number of historical days to retrieve
        end_date: End date for data retrieval (defaults to today)

    Returns:
        DataFrame with stock data or None if download fails
    """
    try:
        if end_date is None:
            end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=days_back)

        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval="1D",
            progress=False
        )

        if data.empty:
            print(f"No data found for {symbol}")
            return None

        return np.round(data, 2)

    except Exception as e:
        print(f"Error downloading data for {symbol}: {str(e)}")
        return None


def find_price_extremes(data: pd.DataFrame) -> Tuple[dict, dict]:
    """
    Find the highest and lowest prices and their positions in the dataset.

    Args:
        data: DataFrame with stock price data

    Returns:
        Tuple of dictionaries containing information about price extremes
    """
    low_info = {
        'price': data["Low"].min(),
        'date': data["Low"].idxmin(),
        'position': data.index.get_loc(data["Low"].idxmin())
    }

    high_info = {
        'price': data["High"].max(),
        'date': data["High"].idxmax(),
        'position': data.index.get_loc(data["High"].idxmax())
    }

    return low_info, high_info


def calculate_trendlines(
        data: pd.DataFrame,
        low_info: dict,
        high_info: dict
) -> List[List[Tuple]]:
    """
    Calculate points for trendlines connecting price extremes to most recent price.

    Args:
        data: DataFrame with stock price data
        low_info: Dictionary with low price information
        high_info: Dictionary with high price information

    Returns:
        List of points defining the trendlines
    """
    data_length = len(data)

    # Calculate days from end for each extreme point
    days_from_low = data_length - low_info['position']
    days_from_high = data_length - high_info['position']

    # Define trendline points
    return [
        [
            (data.index[-days_from_low], data["Low"].iloc[-days_from_low]),
            (data.index[-1], data["Low"].iloc[-1])
        ],
        [
            (data.index[-days_from_high], data["High"].iloc[-days_from_high]),
            (data.index[-1], data["High"].iloc[-1])
        ]
    ]


def plot_stock_chart(
        data: pd.DataFrame,
        symbol: str,
        trendlines: List[List[Tuple]],
        show_volume: bool = True,
        figsize: Tuple[int, int] = (16, 8)
) -> None:
    """
    Create and display a candlestick chart with trendlines.

    Args:
        data: DataFrame with stock price data
        symbol: Stock ticker symbol
        trendlines: List of points defining the trendlines
        show_volume: Whether to display volume
        figsize: Figure dimensions (width, height)
    """
    plt.figure(figsize=figsize)

    mpf.plot(
        data,
        type='candle',
        style='charles',
        alines=dict(
            alines=trendlines,
            colors=['green', 'red'],
            linewidths=1.5,
            alpha=0.7
        ),
        volume=show_volume,
        title=f'{symbol} Stock Price with Trendlines',
        ylabel='Price',
        ylabel_lower='Volume' if show_volume else '',
        figsize=figsize
    )

    plt.show()


def main():
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Configuration
    SYMBOL = "AAPL"
    DAYS_BACK = 400
    SHOW_VOLUME = True

    # Get stock data
    data = get_stock_data(SYMBOL, DAYS_BACK)
    if data is None:
        return

    # Find price extremes
    low_info, high_info = find_price_extremes(data)

    # Calculate trendlines
    trendlines = calculate_trendlines(data, low_info, high_info)

    # Create and display the chart
    plot_stock_chart(data, SYMBOL, trendlines, SHOW_VOLUME)

    # Print summary statistics
    print(f"\nSummary for {SYMBOL}:")
    print(f"Lowest price: ${low_info['price']:.2f} on {low_info['date'].strftime('%Y-%m-%d')}")
    print(f"Highest price: ${high_info['price']:.2f} on {high_info['date'].strftime('%Y-%m-%d')}")
    print(f"Current price: ${data['Close'][-1]:.2f}")


if __name__ == "__main__":
    main()
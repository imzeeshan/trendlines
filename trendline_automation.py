import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from dataclasses import dataclass
from typing import Tuple, List, Optional
import mysql.connector
from datetime import datetime
import logging


@dataclass
class TrendlineCoefficients:
    slope: float
    intercept: float


class TrendlineAnalyzer:
    def __init__(self, min_step: float = 0.0001, optimization_step: float = 1.0):
        self.min_step = min_step
        self.optimization_step = optimization_step
        self.logger = logging.getLogger(__name__)

    def check_trend_line(self, support: bool, pivot: int, slope: float,
                         prices: np.array) -> float:
        """
        Compute validity of trend line and return sum of squared differences.

        Args:
            support: True if checking support line, False if resistance
            pivot: Index of pivot point
            slope: Slope of the line
            prices: Array of price data

        Returns:
            float: Sum of squared differences if valid, -1.0 if invalid
        """
        intercept = -slope * pivot + prices[pivot]
        line_vals = slope * np.arange(len(prices)) + intercept
        diffs = line_vals - prices

        # Validate line position relative to prices
        if (support and diffs.max() > 1e-5) or (not support and diffs.min() < -1e-5):
            return -1.0

        return np.sum(diffs ** 2.0)

    def optimize_slope(self, support: bool, pivot: int, init_slope: float,
                       prices: np.array) -> TrendlineCoefficients:
        """
        Optimize the slope of the trend line using numerical optimization.

        Args:
            support: True if optimizing support line, False if resistance
            pivot: Index of pivot point
            init_slope: Initial slope guess
            prices: Array of price data

        Returns:
            TrendlineCoefficients: Optimized slope and intercept
        """
        slope_unit = (prices.max() - prices.min()) / len(prices)
        curr_step = self.optimization_step
        best_slope = init_slope
        best_err = self.check_trend_line(support, pivot, init_slope, prices)

        if best_err < 0:
            raise ValueError("Invalid initial slope")

        while curr_step > self.min_step:
            # Calculate derivative numerically
            derivative = self._calculate_derivative(support, pivot, best_slope,
                                                    best_err, slope_unit, prices)
            if derivative is None:
                break

            # Test new slope
            test_slope = best_slope + (-slope_unit * curr_step if derivative > 0
                                       else slope_unit * curr_step)
            test_err = self.check_trend_line(support, pivot, test_slope, prices)

            if test_err < 0 or test_err >= best_err:
                curr_step *= 0.5
            else:
                best_err = test_err
                best_slope = test_slope

        return TrendlineCoefficients(
            slope=best_slope,
            intercept=-best_slope * pivot + prices[pivot]
        )

    def _calculate_derivative(self, support: bool, pivot: int, slope: float,
                              error: float, slope_unit: float,
                              prices: np.array) -> Optional[float]:
        """Calculate numerical derivative for slope optimization."""
        for direction in [1, -1]:
            test_slope = slope + direction * slope_unit * self.min_step
            test_err = self.check_trend_line(support, pivot, test_slope, prices)

            if test_err >= 0:
                return direction * (test_err - error)

        self.logger.warning("Derivative calculation failed")
        return None

    def fit_trendlines_high_low(self, high: np.array, low: np.array,
                                close: np.array) -> Tuple[TrendlineCoefficients,
    TrendlineCoefficients]:
        """Fit trendlines using high/low prices."""
        x = np.arange(len(close))
        init_coefs = np.polyfit(x, close, 1)
        line_points = init_coefs[0] * x + init_coefs[1]

        upper_pivot = (high - line_points).argmax()
        lower_pivot = (low - line_points).argmin()

        support_coefs = self.optimize_slope(True, lower_pivot,
                                            init_coefs[0], low)
        resist_coefs = self.optimize_slope(False, upper_pivot,
                                           init_coefs[0], high)

        return support_coefs, resist_coefs


class MarketDataAnalyzer:
    def __init__(self, connection_params: dict, lookback: int = 2000):
        self.connection_params = connection_params
        self.lookback = lookback
        self.analyzer = TrendlineAnalyzer()

    def fetch_data(self, symbol: str, start_date: str) -> pd.DataFrame:
        """Fetch and prepare market data."""
        try:
            connection = mysql.connector.connect(**self.connection_params)
            query = """
                SELECT date, open, high, low, close 
                FROM futures_price 
                WHERE date >= %s AND symbol = %s
            """
            data = pd.read_sql_query(query, connection, params=(start_date, symbol))
            connection.close()

            data = data.set_index('date')
            return np.log(data)  # Apply log transform
        except Exception as e:
            logging.error(f"Database error: {str(e)}")
            raise
        finally:
            if 'connection' in locals() and connection.is_connected():
                connection.close()

    def calculate_trendlines(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate support and resistance trendlines."""
        support_slope = [np.nan] * len(data)
        resist_slope = [np.nan] * len(data)

        for i in range(self.lookback - 1, len(data)):
            candles = data.iloc[i - self.lookback + 1: i + 1]
            support_coefs, resist_coefs = self.analyzer.fit_trendlines_high_low(
                candles['high'].values,
                candles['low'].values,
                candles['close'].values
            )
            support_slope[i] = support_coefs.slope
            resist_slope[i] = resist_coefs.slope

        data['support_slope'] = support_slope
        data['resist_slope'] = resist_slope
        return data


class TrendlinePlotter:
    @staticmethod
    def plot_trendlines(data: pd.DataFrame, lookback: int):
        """Plot candlestick chart with trendlines."""
        candles = data.iloc[-lookback:]
        analyzer = TrendlineAnalyzer()

        # Calculate trendlines
        support_line, resist_line = TrendlinePlotter._calculate_plot_lines(
            candles, analyzer)

        # Setup plot
        plt.style.use('dark_background')
        ax = plt.gca()

        # Convert line points to mplfinance format
        s_seq = TrendlinePlotter._get_line_points(candles, support_line)
        r_seq = TrendlinePlotter._get_line_points(candles, resist_line)

        # Plot
        mpf.plot(candles,
                 alines=dict(alines=[s_seq, r_seq],
                             colors=['w', 'w']),
                 type='candle',
                 style='charles',
                 ax=ax)
        plt.show()

    @staticmethod
    def _calculate_plot_lines(candles: pd.DataFrame,
                              analyzer: TrendlineAnalyzer) -> Tuple[np.array, np.array]:
        """Calculate support and resistance lines for plotting."""
        x = np.arange(len(candles))
        support_coefs, resist_coefs = analyzer.fit_trendlines_high_low(
            candles['high'].values,
            candles['low'].values,
            candles['close'].values
        )

        support_line = support_coefs.slope * x + support_coefs.intercept
        resist_line = resist_coefs.slope * x + resist_coefs.intercept

        return support_line, resist_line

    @staticmethod
    def _get_line_points(candles: pd.DataFrame,
                         line_points: np.array) -> List[Tuple[datetime, float]]:
        """Convert line points to mplfinance format."""
        idx = candles.index
        return [(idx[i], line_points[i]) for i in range(len(line_points))]


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Configuration
    connection_params = {
        "host": "localhost",
        "port": 3306,
        "user": "zeeshan",
        "password": "zeeshan1983",
        "database": "lucrumbot"
    }

    # Initialize analyzer
    market_analyzer = MarketDataAnalyzer(connection_params)

    # Fetch and analyze data
    data = market_analyzer.fetch_data("NQ", "2025-01-16 18:00:00")
    analyzed_data = market_analyzer.calculate_trendlines(data)

    # Plot results
    TrendlinePlotter.plot_trendlines(analyzed_data, market_analyzer.lookback)


if __name__ == "__main__":
    main()
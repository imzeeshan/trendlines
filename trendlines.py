import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import mysql.connector

connection = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="zeeshan",
    password="zeeshan1983",
    database="lucrumbot"
)

global_slope_up = 0.01
global_slope_down = -0.01
threshold_value=0.000000000001
PEAK, VALLEY = 1, -1
extend_by = 2

def plot_candlestick_with_trends(df, trends):
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],

                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])

    if not trends.empty:
        for _, row in trends.iterrows():
            trend_color = 'green' if row['Trend'] == 'Up' else 'red'
            fig.add_trace(go.Scatter(
                x=[row['Start Date'], row['End Date']],
                y=[row['Start Price'], row['End Price']],
                mode='lines',
                line=dict(color=trend_color, width=2),
                name=f"{row['Trend']} Trend"
            ))

    fig.update_layout(
        title=f"Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,

    )

    fig.show()


def plot_candlestick_with_unfiltered_trends(df, trends):
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],

                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])

    # Add trendlines based on trends DataFrame
    if not trends.empty:
        for _, row in trends.iterrows():
            trend_color = 'green'
            fig.add_trace(go.Scatter(
                x=[row['Start Timeframe'], row['End Timeframe']],
                y=[row['Start Price'], row['End Price']],
                mode='lines',
                line=dict(color=trend_color, width=2),
            ))

    # Update layout for better visuals
    fig.update_layout(
        title=f"Candlestick Chart with Trendlines for {tickerSymbol}",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,

    )

    # Show the plot
    fig.show()


def _identify_initial_pivot(X, up_thresh, down_thresh):
    x_0 = X[0]
    max_x = x_0
    max_t = 0
    min_x = x_0
    min_t = 0
    up_thresh += 1
    down_thresh += 1

    for t in range(1, len(X)):
        x_t = X[t]

        if x_t / min_x >= up_thresh:
            return VALLEY if min_t == 0 else PEAK

        if x_t / max_x <= down_thresh:
            return PEAK if max_t == 0 else VALLEY

        if x_t > max_x:
            max_x = x_t
            max_t = t

        if x_t < min_x:
            min_x = x_t
            min_t = t

    t_n = len(X)-1
    return VALLEY if x_0 < X[t_n] else PEAK

def calculate_pivot_points(close, high, low, up_thresh, down_thresh):


    initial_pivot = _identify_initial_pivot(close, up_thresh, down_thresh)

    t_n = len(close)
    pivots = np.zeros(t_n, dtype='i1')
    pivots[0] = initial_pivot

    up_thresh += 1
    down_thresh += 1

    trend = -initial_pivot
    last_pivot_t = 0
    last_pivot_x = close[0]
    for t in range(1, len(close)):

        if trend == -1:
            x = low[t]
            r = x / last_pivot_x
            if r >= up_thresh:
                pivots[last_pivot_t] = trend
                trend = 1
                last_pivot_x = high[t]
                last_pivot_t = t
            elif x < last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t
        else:
            x = high[t]
            r = x / last_pivot_x
            if r <= down_thresh:
                pivots[last_pivot_t] = trend
                trend = -1
                last_pivot_x = low[t]
                last_pivot_t = t
            elif x > last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t


    if last_pivot_t == t_n-1:
        pivots[last_pivot_t] = trend
    elif pivots[t_n-1] == 0:
        pivots[t_n-1] = trend

    return pivots

# %%
def find_high_low_type(df):
    prev_high = float('-inf')
    prev_low = float('inf')
    high_patterns = []
    low_patterns = []

    for index, row in df.iterrows():
        high = row['High']
        low = row['Low']
        pivot = row['Pivots']

        if pivot == 1:
            if high > prev_high:
                high_pattern = "HH"
                low_pattern = None
                prev_high = high
            else:
                high_pattern = "LH"
                low_pattern = None
                prev_high = high
        elif pivot == -1:
            if low > prev_low:
                high_pattern = None
                low_pattern = "HL"
                prev_low = low
            else:
                high_pattern = None
                low_pattern = "LL"
                prev_low = low

        else:
            high_pattern = None
            low_pattern = None

        high_patterns.append(high_pattern)
        low_patterns.append(low_pattern)

    df['High_pattern'] = high_patterns
    df['Low_pattern'] = low_patterns

def find_downtrend(df):
    df=df.reset_index()
    df['Trend_Down'] = None
    for i in range(0, len(df)-2):
        if df['High_pattern'].iloc[i] == 'HH' and df['Peaks'].iloc[i] > df['Peaks'].iloc[i+1] and df['Peaks'].iloc[i+1] > df['Peaks'].iloc[i+2]:
            df.at[i, 'Trend_Down'] = 'down'
    for i in range(1, len(df)-2):
        if df['Peaks'].iloc[i] > df['Peaks'].iloc[i-1] and df['Peaks'].iloc[i] > df['Peaks'].iloc[i+1] and df['Peaks'].iloc[i+1] > df['Peaks'].iloc[i+2]:
            df.at[i, 'Trend_Down'] = 'down'
        if df['High_pattern'].iloc[i] == 'LH' and df['High_pattern'].iloc[i-1] == 'HH' and df['High_pattern'].iloc[i+1] == 'LH':
            df.at[i, 'Trend_Down'] = 'down'
    for i in range(1, len(df)):
        if df['Trend_Down'].iloc[i-1] == 'down' and df['High_pattern'].iloc[i] == 'LH':
            df.at[i, 'Trend_Down'] = 'down'    
    return df

def find_uptrend(df):
    df=df.reset_index()
    df['Trend_Up'] = None
    for i in range(0, len(df)-2):
        if df['Low_pattern'].iloc[i] == 'LL' and df['Troughs'].iloc[i] < df['Troughs'].iloc[i+1] and df['Troughs'].iloc[i+1] < df['Troughs'].iloc[i+2]:
            df.at[i, 'Trend_Up'] = 'up'
    for i in range(1, len(df)-2):
        if df['Troughs'].iloc[i] < df['Troughs'].iloc[i-1] and df['Troughs'].iloc[i] < df['Troughs'].iloc[i+1] and df['Troughs'].iloc[i+1] < df['Troughs'].iloc[i+2]:
            df.at[i, 'Trend_Up'] = 'up'
        if df['Low_pattern'].iloc[i] == 'HL' and df['Low_pattern'].iloc[i-1] == 'LL' and df['Low_pattern'].iloc[i+1] == 'HL':
            df.at[i, 'Trend_Up'] = 'up'
    for i in range(1, len(df)):
        if df['Trend_Up'].iloc[i-1] == 'up' and df['Low_pattern'].iloc[i] == 'HL':
            df.at[i, 'Trend_Up'] = 'up'    
    return df

def find_consecutive_up_labels(df_low):
    line_up_data = []
    for i in range(0, len(df_low)-1):
        if df_low['Trend_Up'].iloc[i] == 'up' and df_low['Trend_Up'].iloc[i+1] == 'up':
            start_timeframe = df_low['Date'].iloc[i]
            start_price = df_low['Troughs'].iloc[i]
            end_timeframe = df_low['Date'].iloc[i+1]
            end_price = df_low['Troughs'].iloc[i+1]
            line_up_data.append([start_timeframe, start_price, end_timeframe, end_price])

    line_up = pd.DataFrame(line_up_data, columns=['Start Timeframe', 'Start Price', 'End Timeframe', 'End Price'])

    return line_up

# %%
def find_consecutive_down_labels(df_high):
    line_down_data = []  # Collect data in lists

    for i in range(0, len(df_high) - 1):
        if df_high['Trend_Down'].iloc[i] == 'down' and df_high['Trend_Down'].iloc[i + 1] == 'down':
            start_timeframe = df_high['Date'].iloc[i]
            start_price = df_high['Peaks'].iloc[i]
            end_timeframe = df_high['Date'].iloc[i + 1]
            end_price = df_high['Peaks'].iloc[i + 1]
            line_down_data.append([start_timeframe, start_price, end_timeframe, end_price])

    line_down = pd.DataFrame(line_down_data, columns=['Start Timeframe', 'Start Price', 'End Timeframe', 'End Price'])

    return line_down

# %%
def final_uptrend(df, x_up, y_up):
    dates_up = pd.to_datetime(x_up)

    if len(dates_up) > 0 and len(x_up) == len(y_up):

        price_up = np.array(y_up)
        t = ((dates_up - dates_up[0]).total_seconds() / 60).values
        # t = (dates_up - dates_up[0]).days.values
        time_frame_up = t.reshape(-1, 1)
        model = LinearRegression()
        model.fit(time_frame_up, price_up)

        slope = model.coef_[0]
        c = []

        for i in range(0, len(dates_up) - 1):
            c.append(price_up[i] - (slope * time_frame_up[i]))

        min_c = 100000000000

        for i in range(0, len(c)):
            if c[i] < min_c:
                min_c = c[i]

        price_final_up = []

        for i in range(0, len(dates_up)):
            price_final_up.append((slope * time_frame_up[i]) + min_c)

        price_final_flat_up = np.concatenate(price_final_up, axis=0)

        d_line_up = pd.DataFrame({'Date': dates_up, 'Predicted Price': price_final_flat_up})
        up_trend_values = pd.DataFrame({
            'Start Date': d_line_up['Date'].iloc[0],
            'Start Price': d_line_up['Predicted Price'].iloc[0],
            'End Date': d_line_up['Date'].iloc[-1],
            'End Price': d_line_up['Predicted Price'].iloc[-1],
            'Trend': ['Up'],
            'Slope': slope,
            'Intercept': min_c,
        })
        return up_trend_values
    else:
        print('Could not detect any up trend lines')

# %%
def final_downtrend(df, x_down, y_down):
    dates_down = pd.to_datetime(x_down)

    if len(dates_down) > 0 and len(x_down) == len(y_down):

        price_down = np.array(y_down)
        t = ((dates_down - dates_down[0]).total_seconds() / 60).values
        # t = (dates_down - dates_down[0]).days.values

        time_frame_down = t.reshape(-1, 1)
        model = LinearRegression()
        model.fit(time_frame_down, price_down)

        slope = model.coef_[0]

        c = []

        for i in range(0, len(dates_down) - 1):
            c.append(price_down[i] - (slope * time_frame_down[i]))

        max_c = -100000000000

        for i in range(0, len(c)):
            if c[i] > max_c:
                max_c = c[i]

        price_final_down = []

        for i in range(0, len(dates_down)):
            price_final_down.append((slope * time_frame_down[i]) + max_c)

        price_final_flat_down = np.concatenate(price_final_down, axis=0)

        d_line_down = pd.DataFrame({'Date': dates_down, 'Predicted Price': price_final_flat_down})

        down_trend_values = pd.DataFrame({
            'Start Date': d_line_down['Date'].iloc[0],
            'Start Price': d_line_down['Predicted Price'].iloc[0],
            'End Date': d_line_down['Date'].iloc[-1],
            'End Price': d_line_down['Predicted Price'].iloc[-1],
            'Trend': ['Down'],
            'Slope': slope,
            'Intercept': max_c
        })
        return down_trend_values
    else:
        print('Could not detect any down trend lines')
    
    
# %%
def separate_group(df, trend_df, trend_type):
    consecutive_groups = []
    current_group = {'x': [], 'y': []}
    final_trend = pd.DataFrame()
    for i in range(0, len(trend_df) - 1):
        if trend_df['Start Timeframe'].iloc[i] not in current_group['x']:
            current_group['x'].append(trend_df['Start Timeframe'].iloc[i])
        if trend_df['End Timeframe'].iloc[i] not in current_group['x']:
            current_group['x'].append(trend_df['End Timeframe'].iloc[i])
        if trend_df['Start Price'].iloc[i] not in current_group['y']:
            current_group['y'].append(trend_df['Start Price'].iloc[i])
        if trend_df['End Price'].iloc[i] not in current_group['y']:
            current_group['y'].append(trend_df['End Price'].iloc[i])
        if trend_df.iloc[i]["End Timeframe"] != trend_df.iloc[i + 1]["Start Timeframe"]:
            consecutive_groups.append(current_group)
            current_group = {'x': [], 'y': []}
    current_group['x'].extend([trend_df['Start Timeframe'].iloc[-1], trend_df['End Timeframe'].iloc[-1]])
    current_group['y'].extend([trend_df['Start Price'].iloc[-1], trend_df['End Price'].iloc[-1]])
    consecutive_groups.append(current_group)
    for group in consecutive_groups:
        if (trend_type == 'up'):
            trend_value = final_uptrend(df, group['x'], group['y'])
        else:
            trend_value = final_downtrend(df, group['x'], group['y'])
        final_trend = pd.concat([final_trend, trend_value])

    return final_trend


def initialize_df(df):
    pivots = calculate_pivot_points(df.Close, df.High, df.Low, threshold_value, -1 * threshold_value)

    df['Pivots'] = pivots
    df['Pivot Price'] = np.nan
    df['Peaks'] = np.nan
    df['Troughs'] = np.nan
    trend_value = pd.DataFrame()

    df.loc[df['Pivots'] == 1, 'Pivot Price'] = df.High
    df.loc[df['Pivots'] == -1, 'Pivot Price'] = df.Low
    df.loc[df['Pivots'] == 1, 'Peaks'] = df.High
    df.loc[df['Pivots'] == -1, 'Troughs'] = df.Low
    find_high_low_type(df)

    df_high = df.dropna(subset=['High_pattern'])
    df_low = df.dropna(subset=['Low_pattern'])

    df_high = find_downtrend(df_high)
    df_low = find_uptrend(df_low)

    uptrend = find_consecutive_up_labels(df_low)
    downtrend = find_consecutive_down_labels(df_high)

    unfiltered_trend = pd.concat([uptrend, downtrend])

    if len(uptrend) > 0:
        trend_value = pd.concat([trend_value, separate_group(df, uptrend, 'up')])
    if len(downtrend) > 0:
        trend_value = pd.concat([trend_value, separate_group(df, downtrend, 'down')])
    return trend_value, unfiltered_trend

def calculate_extended_trends(df, trends):
    for j in range(len(df)):
        if trends['End Date'] == df['Date'].iloc[j]:
            date_start = pd.to_datetime(df['Date'].iloc[j])
            date_end = pd.to_datetime(df['Date'].iloc[j + extend_by])
            x = ((date_end - date_start).total_seconds()/60)
            y = trends['Slope'] * x + trends['End Price']
            trend_new = pd.DataFrame({
                'Start Date': [trends['Start Date']],
                'Start Price': [trends['Start Price']],
                'End Date': [df['Date'].iloc[j + extend_by]],
                'End Price': [y],
                'Trend': [trends['Trend']],
            })
            return trend_new

df = pd.read_sql_query("select date as Date ,open as Open,high  as High,low as Low ,close as Close "
                       "from futures_price_1min where date >= '2025-01-14 18:00:00' and symbol='GC'", con=connection)
trends, unfiltered_trend = initialize_df(df)
# print(trends)
df['Date'] = pd.to_datetime(df['Date'])

extended_trend = pd.DataFrame()
for i in range(len(trends)):
    output_trend = calculate_extended_trends(df, trends.iloc[i])
    extended_trend = pd.concat([extended_trend, output_trend], ignore_index=True)

df['Date'] = pd.to_datetime(df['Date'])
plot_candlestick_with_trends(df, extended_trend)
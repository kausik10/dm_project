import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def plot_future_projection(stock_data, symbol, lookback_days=30, forward_days=10):
    """
    Projects next N trading days of price using linear regression based on recent trend.

    Args:
        stock_data (pd.DataFrame): Historical stock data sorted by date.
        symbol (str): Stock symbol.
        lookback_days (int): Days to look back for regression fitting.
        forward_days (int): Days to project into the future.

    Returns:
        fig (go.Figure): Plotly figure showing past and projected future prices.
    """
    stock_data = stock_data.sort_values('Date')
    recent_data = stock_data[-lookback_days:].copy().reset_index(drop=True)

    # Fit linear regression on lookback window
    X = np.arange(lookback_days).reshape(-1, 1)
    y = recent_data['ClosePrice'].values.reshape(-1, 1)

    model = LinearRegression().fit(X, y)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    # Predict future
    X_future = np.arange(lookback_days, lookback_days + forward_days).reshape(-1, 1)
    y_future = model.predict(X_future)

    future_dates = pd.date_range(start=stock_data['Date'].iloc[-1], periods=forward_days + 1, freq='B')[1:]

    # Plot
    fig = go.Figure()

    # Recent actual prices
    fig.add_trace(go.Scatter(
        x=recent_data['Date'],
        y=recent_data['ClosePrice'],
        mode='lines+markers',
        name='Recent Close Price',
        line=dict(color='lightblue')
    ))

    # Projected prices
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=y_future.flatten(),
        mode='lines+markers',
        name='Projected Next 10 Days',
        line=dict(color='orange', dash='dash')
    ))

    fig.update_layout(
        title=f"{symbol} - 10-Day Price Projection",
        xaxis_title="Date",
        yaxis_title="Projected Close Price",
        template='plotly_dark',
        height=600,
        width=1000
    )

    return fig, future_dates, y_future.flatten()

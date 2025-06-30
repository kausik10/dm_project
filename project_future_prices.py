import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # Import PolynomialFeatures

def plot_future_projection(stock_data, symbol, lookback_days=30, forward_days=10):
    """
    Plots future stock price projections using polynomial regression for a smoother curve.

    Args:
        stock_data (pd.DataFrame): DataFrame with 'Date' and 'ClosePrice' columns.
        symbol (str): Stock symbol.
        lookback_days (int): Number of past days to consider for training.
        forward_days (int): Number of future days to project.

    Returns:
        tuple: (fig, future_dates, y_future)
            fig (go.Figure): Plotly figure.
            future_dates (pd.Series): Series of projected future dates.
            y_future (np.ndarray): Array of projected future prices.
    """
    stock_data = stock_data.sort_values('Date').copy()
    recent_data = stock_data[-lookback_days:].reset_index(drop=True)

    # Create numerical X based on actual date offsets (in days)
    recent_data['DayIndex'] = (recent_data['Date'] - recent_data['Date'].min()).dt.days
    X = recent_data['DayIndex'].values.reshape(-1, 1)
    y = recent_data['ClosePrice'].values.reshape(-1, 1)

    # --- Changes start here: Implement Polynomial Regression ---
    # You can adjust the 'degree' based on how much curve you want.
    # A degree of 2 is often a good starting point for a smoother curve than linear.
    # Higher degrees can lead to overfitting, so use with caution.
    degree = 2 # Default polynomial degree. Can be adjusted.

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Fit model using polynomial features
    model = LinearRegression().fit(X_poly, y)

    # Prepare future X
    last_day_index = X[-1][0]
    future_day_indices = np.arange(last_day_index + 1, last_day_index + forward_days + 1).reshape(-1, 1)
    
    # Transform future X for prediction using the same polynomial features
    future_day_indices_poly = poly.transform(future_day_indices)
    
    # Predict future prices
    y_future = model.predict(future_day_indices_poly)
    # --- Changes end here ---

    # Generate future business dates
    future_dates = pd.date_range(start=recent_data['Date'].iloc[-1], periods=forward_days + 1, freq='B')[1:]

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recent_data['Date'],
        y=recent_data['ClosePrice'],
        mode='lines+markers',
        name='Recent Close Price',
        line=dict(color='lightblue')
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=y_future.flatten(),
        mode='lines+markers',
        name=f'Projected Next {forward_days} Days (Poly Degree {degree})', # Updated name
        line=dict(color='orange', dash='dash')
    ))

    # Add the fitted polynomial curve for the recent data to show the fit
    # This helps visualize how well the chosen degree fits the historical data
    X_plot_recent = np.arange(recent_data['DayIndex'].min(), recent_data['DayIndex'].max() + 1).reshape(-1, 1)
    X_plot_recent_poly = poly.transform(X_plot_recent)
    y_plot_recent_poly = model.predict(X_plot_recent_poly)
    
    fig.add_trace(go.Scatter(
        x=recent_data['Date'].min() + pd.to_timedelta(X_plot_recent.flatten(), unit='D'),
        y=y_plot_recent_poly.flatten(),
        mode='lines',
        name=f'Polynomial Fit (Degree {degree})',
        line=dict(color='green', dash='dot')
    ))


    fig.update_layout(
        title=f"{symbol} - {forward_days}-Day Price Projection (Polynomial Regression)", # Updated title
        xaxis_title="Date",
        yaxis_title="Projected Close Price",
        template='plotly_dark',
        height=600,
        width=1000,
        hovermode='x unified' # Added for better interactivity
    )

    return fig, future_dates, y_future.flatten()

# Note: The example usage block from the previous response is removed
# to ensure the function matches your original request for direct integration.
# You can test this by running your main.py with this updated function definition.
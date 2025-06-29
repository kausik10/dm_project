import plotly.graph_objects as go

def plot_candlestick(nepse_combined_df, symbol):
    """
    Generate a candlestick chart for a single stock symbol.

    Args:
        nepse_combined_df (pd.DataFrame): DataFrame containing stock data.
        symbol (str): Stock symbol to plot.

    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure object for the candlestick chart.
    """
    # Filter data for the selected symbol and sort by Date
    stock_data = nepse_combined_df[nepse_combined_df['Symbol'] == symbol].sort_values('Date')

    if stock_data.empty:
        print(f"No data found for {symbol}")
        return None

    fig = go.Figure(data=[go.Candlestick(
        x=stock_data['Date'],
        open=stock_data['OpenPrice'],
        high=stock_data['HighPrice'],
        low=stock_data['LowPrice'],
        close=stock_data['ClosePrice'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    fig.update_layout(
        title=f'Candlestick Chart for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=600,
        width=1000
    )

    return fig

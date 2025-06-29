import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_stock_price(df):
    """
    Calculate correlation matrix of stock closing prices and plot heatmap.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['Date', 'Symbol', 'ClosePrice']
    
    Returns:
        corr_matrix (pd.DataFrame): correlation matrix of closing prices
        fig (matplotlib.figure.Figure): heatmap figure object
    """
    # Pivot table: dates as rows, symbols as columns, values = close prices
    pivot_df = df.pivot_table(index='Date', columns='Symbol', values='ClosePrice')
    
    # Calculate correlation matrix (pairwise correlation of stocks)
    corr_matrix = pivot_df.corr()
    
    # Create figure for heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', ax=ax, annot=False)
    ax.set_title('Stock Closing Price Correlation Matrix')
    
    # Return correlation matrix and figure
    return corr_matrix, fig

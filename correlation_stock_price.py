import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_stock_price(df):
    pivot_df = df.pivot_table(index='Date', columns='Symbol', values='ClosePrice')
    
    corr_matrix = pivot_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', ax=ax, annot=False)
    ax.set_title('Stock Closing Price Correlation Matrix')
    
    return corr_matrix, fig

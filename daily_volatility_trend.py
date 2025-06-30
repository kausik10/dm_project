import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_volatility_trend(df, symbols):
    # Ensure Volatility is calculated
    if 'Volatility' not in df.columns:
        df['Volatility'] = df['HighPrice'] - df['LowPrice']
    
    # Handle single string input
    if isinstance(symbols, str):
        symbols = [symbols]

    # Filter data for requested symbols
    filtered_df = df[df['Symbol'].isin(symbols)].sort_values(['Symbol', 'Date'])

    if filtered_df.empty:
        st.warning("⚠️ No data found for the given symbol(s).")
        return None

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=filtered_df, x='Date', y='Volatility', hue='Symbol')
    plt.title(f"Daily Volatility Trend - {', '.join(symbols)}")
    plt.xlabel('Date')
    plt.ylabel('Volatility (High - Low)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    return plt.gcf()

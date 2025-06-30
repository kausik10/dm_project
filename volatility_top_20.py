import matplotlib.pyplot as plt

def calculate_volatility_plot(df, top_n=20):
    df['Volatility'] = df['HighPrice'] - df['LowPrice']

    # Group by symbol and calculate average volatility
    volatility_by_symbol = (
        df.groupby('Symbol')['Volatility']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    volatility_by_symbol.plot(kind='bar', ax=ax, color='skyblue', title='Average Volatility of Top 20 Symbols')
    ax.set_ylabel('Average Daily Volatility')
    ax.grid(True)

    for i, value in enumerate(volatility_by_symbol):
        ax.text(i, value + 0.01, f'{value:.2f}', ha='center', va='bottom')

    return fig
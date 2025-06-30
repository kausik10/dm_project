import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

def plot_closing_price_trend(nepse_combined_df, company_symbols=['BHL']):
    if isinstance(company_symbols, str):
        company_symbols = [company_symbols]

    available_symbols = nepse_combined_df['Symbol'].unique()
    valid_symbols = [s for s in company_symbols if s in available_symbols]

    if not valid_symbols:
        st.warning("None of the selected symbols are available in the dataset.")
        return

    df_plot = nepse_combined_df[nepse_combined_df['Symbol'].isin(valid_symbols)].copy()
    df_plot = df_plot.sort_values('Date')

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_plot, x='Date', y='ClosePrice', hue='Symbol')
    plt.title('Closing Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.grid(True)
    plt.legend(title='Symbol')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

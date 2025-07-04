import streamlit as st
import pandas as pd
from volatility_top_20 import calculate_volatility_plot
from heatmap_by_sector import plot_stock_heatmap
from daily_volatility_trend import plot_volatility_trend
from stock_clusters import plot_stock_clusters
from closing_price_trend import plot_closing_price_trend
from correlation_stock_price import correlation_stock_price
from individual_candle_stick import plot_candlestick
from stock_future_trend import detect_trend, get_price_trend_slope
from project_future_prices import plot_future_projection

st.set_page_config(page_title="NEPSE Dashboard", layout="wide")
st.header("📈 NEPSE Dashboard")
@st.cache_data
def load_data():
    # Load CSVs
    nepse_combined_df = pd.read_csv("nepse_combined_cleaned.csv", low_memory=False)
    nepse_index_df = pd.read_csv("combined_nepse_index_data.csv")

    # Process 'Date' column
    nepse_index_df.rename(columns={'Date (AD)': 'Date'}, inplace=True)
    nepse_index_df['Date'] = nepse_index_df['Date'].astype(str).str.strip()
    nepse_index_df['Date'] = pd.to_datetime(nepse_index_df['Date'], format='mixed', errors='coerce')
    nepse_index_df.dropna(subset=['Date'], inplace=True)
    nepse_index_df = nepse_index_df[nepse_index_df['Date'] >= pd.to_datetime('2024-03-04')].copy()
    nepse_index_df.sort_values('Date', inplace=True)
    nepse_index_df['Date'] = nepse_index_df['Date'].dt.strftime('%Y-%m-%d')

    nepse_combined_df['Date'] = pd.to_datetime(nepse_combined_df['Date'], format='%Y_%m_%d', errors='coerce')
    nepse_combined_df.dropna(subset=['Date'], inplace=True)

    nepse_combined_df.rename(columns={
        'S.No': 'SerialNo',
        'Symbol': 'Symbol',
        'Conf.': 'Confidence',
        'Open': 'OpenPrice',
        'High': 'HighPrice',
        'Low': 'LowPrice',
        'Close': 'ClosePrice',
        'VWAP': 'VWAP',
        'Vol': 'Volume',
        'Prev. Close': 'PrevClosePrice',
        'Turnover': 'Turnover',
        'Trans.': 'Transactions',
        'Diff': 'Difference',
        'Range': 'RangeValue',
        'Diff %': 'DiffPercent',
        'Range %': 'RangePercent',
        'VWAP %': 'VWAPPercent',
        '120 Days': '120Days',
        '180 Days': '180Days',
        '52 Weeks High': 'High_52Weeks',
        '52 Weeks Low': 'Low_52Weeks',
        'LTP': 'LastTradedPrice',
        'Close - LTP': 'CloseMinusLTP',
        'Close - LTP %': 'CloseMinusLTPPercent',
        'Date': 'Date'
    }, inplace=True)

    # Drop rows where Symbol ends with 'P' or 'PO' (promoter shares)
    nepse_combined_df = nepse_combined_df[~nepse_combined_df['Symbol'].str.endswith(('P', 'PO'))]
    # Drop stocks whose Symbol ends with a number (mutual funds like BOKD86, EBLD86, etc.)
    nepse_combined_df = nepse_combined_df[~nepse_combined_df['Symbol'].str.contains(r'\d$', na=False)]



    # Convert to numeric
    cols_to_numeric = [
        'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'VWAP',
        'Volume', 'PrevClosePrice', 'Turnover', 'Transactions',
        'Difference', 'RangeValue', '120Days', '180Days',
        'High_52Weeks', 'Low_52Weeks', 'LastTradedPrice'
    ]
    nepse_combined_df[cols_to_numeric] = nepse_combined_df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')

    # Drop unnecessary columns
    nepse_combined_df.drop(columns=[
        'Turnover', 'LastTradedPrice', 'CloseMinusLTP', 'CloseMinusLTPPercent', 'Volume'
    ], inplace=True)

    # Drop rows with missing core prices
    nepse_combined_df = nepse_combined_df.dropna(subset=[
        'OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice', 'VWAP'
    ])

    # Fill missing 120/180 day values group-wise
    for col in ['120Days', '180Days']:
        nepse_combined_df[col] = nepse_combined_df.groupby('Symbol')[col].transform(lambda x: x.fillna(x.median()))

    # Sort and forward-fill prices
    nepse_combined_df = nepse_combined_df.sort_values(by=['Symbol', 'Date'])
    nepse_combined_df[['OpenPrice', 'ClosePrice']] = nepse_combined_df.groupby('Symbol')[['OpenPrice', 'ClosePrice']].ffill()

    # Fill Transactions and PrevClosePrice
    nepse_combined_df['Transactions'] = nepse_combined_df['Transactions'].fillna(0)
    nepse_combined_df['PrevClosePrice'] = nepse_combined_df['PrevClosePrice'].fillna(nepse_combined_df['PrevClosePrice'].median())

    # Fill more group-wise medians
    nepse_combined_df['High_52Weeks'] = nepse_combined_df.groupby('Symbol')['High_52Weeks'].transform(lambda x: x.fillna(x.median()))

    # Global median fallback
    for col in ['120Days', '180Days', 'High_52Weeks']:
        nepse_combined_df[col] = nepse_combined_df[col].fillna(nepse_combined_df[col].median())

    # Missing % table
    missing_percent = nepse_combined_df.isnull().mean() * 100
    missing_table = pd.DataFrame({
        'Column': missing_percent.index,
        'Missing %': missing_percent.values
    }).sort_values(by='Missing %', ascending=False).reset_index(drop=True)

    # Drop rows with missing core prices
    nepse_combined_df = nepse_combined_df.dropna(subset=[
        'OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice', 'VWAP'
    ])

    # Sort and forward-fill prices
    nepse_combined_df = nepse_combined_df.sort_values(by=['Symbol', 'Date'])
    nepse_combined_df[['OpenPrice', 'ClosePrice']] = nepse_combined_df.groupby('Symbol')[['OpenPrice', 'ClosePrice']].ffill()

    # Fill missing Transactions with 0 (assuming missing = no transaction)
    nepse_combined_df['Transactions'] = nepse_combined_df['Transactions'].fillna(0)

    # Fill PrevClosePrice with median (can also use group-wise)
    nepse_combined_df['PrevClosePrice'] = nepse_combined_df['PrevClosePrice'].fillna(nepse_combined_df['PrevClosePrice'].median())

    # Fill with group medians
    nepse_combined_df['120Days'] = nepse_combined_df.groupby('Symbol')['120Days'].transform(lambda x: x.fillna(x.median()))
    nepse_combined_df['180Days'] = nepse_combined_df.groupby('Symbol')['180Days'].transform(lambda x: x.fillna(x.median()))
    nepse_combined_df['High_52Weeks'] = nepse_combined_df.groupby('Symbol')['High_52Weeks'].transform(lambda x: x.fillna(x.median()))

    # Fill remaining NaNs with global median
    nepse_combined_df['120Days'] = nepse_combined_df['120Days'].fillna(nepse_combined_df['120Days'].median())
    nepse_combined_df['180Days'] = nepse_combined_df['180Days'].fillna(nepse_combined_df['180Days'].median())
    nepse_combined_df['High_52Weeks'] = nepse_combined_df['High_52Weeks'].fillna(nepse_combined_df['High_52Weeks'].median())


    return nepse_combined_df, nepse_index_df, missing_table


# Load data
nepse_combined_df, nepse_index_df, missing_table = load_data()

tabs = st.tabs([
    "🗃️ Data Overview", 
    "🔺 Top 20 Volatile", 
    "🔥 Sector Heatmap", 
    "📈 Volatility Trend", 
    "🔍 Clustering", 
    "📊 Closing Price Trend", 
    "📉 Correlation Matrix", 
    "🕯️ Candlestick Chart",
    "🔮 Stock Trend Prediction",
])

# 1. Data Overview
with tabs[0]:
    # Display data
    st.subheader("NEPSE Combined Stock Data")
    st.dataframe(nepse_combined_df.head())

    st.subheader("NEPSE Index Data (Filtered from 2024-03-04)")
    st.dataframe(nepse_index_df.head())

    st.subheader("📊 Missing Value Percentage (NEPSE Combined)")
    st.dataframe(missing_table)

with tabs[1]:
    # Plot the Volatility of top 20 symbols
    st.subheader("🔺 Top 20 Most Volatile Symbols")

    fig_vol = calculate_volatility_plot(nepse_combined_df)

    left_col, center_col, right_col = st.columns([1, 3, 1])  
    with center_col:
        st.pyplot(fig_vol)


# 3. Heatmap by Sector
with tabs[2]:
    # HeatMap of NEPSE Stocks by Sector
    st.subheader("🔥 NEPSE Stock Heatmap by Sector")
    fig = plot_stock_heatmap(nepse_combined_df)
    st.plotly_chart(fig, use_container_width=True)


# 4. Volatility Trend
with tabs[3]:
    # Daily Volatility Trend Comparison
    st.subheader("📈 Volatility Trend Comparison")

    selected_symbols = st.multiselect("Select Symbols", sorted(nepse_combined_df['Symbol'].unique()), default=['BHL'])

    if selected_symbols:
        fig_vol_trend = plot_volatility_trend(nepse_combined_df, selected_symbols)
        if fig_vol_trend:
            left_col, center_col, right_col = st.columns([1, 3, 1])  
            with center_col:
                    st.pyplot(fig_vol_trend)


# Clustering and PCA Visualization

# 5. Clustering
with tabs[4]:
    # Cluster feature set to be used externally
    cluster_features = [
        'DiffPercent',
        'RangePercent',
        'VWAPPercent',
        'Volatility',
        'Transactions',
        'ClosePrice'
    ]

    st.subheader("🔍 NEPSE Stock Clustering")

    k_val = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=3)
    use_std = st.checkbox("Include Std Deviation of Close Price")

    from stock_clusters import plot_stock_clusters  # local import is fine here

    fig_cluster, df_cluster = plot_stock_clusters(nepse_combined_df, cluster_features.copy(), k=k_val, use_std=use_std)
    left_col, center_col, right_col = st.columns([1, 3, 1])  
    with center_col:
        st.pyplot(fig_cluster)

    # Dispaly the Stocks Based on Clusters
    st.subheader("📌 Stocks per Cluster")
    for cluster_id in sorted(df_cluster['Cluster'].unique()):
        symbols = df_cluster[df_cluster['Cluster'] == cluster_id]['Symbol'].tolist()
        with st.expander(f"Cluster {cluster_id} ({len(symbols)} stocks)"):
            st.write(', '.join(symbols))

# 6. Closing Price Trend
with tabs[5]:
    # Display the closing price trend for selected symbols
    st.subheader("📈 Closing Price Trend")

    selected_symbols = st.multiselect(
        "Select Stocks",
        options=sorted(nepse_combined_df['Symbol'].unique()),
        default=['BHL'],
        key='close_price_trend'
    )
    left_col, center_col, right_col = st.columns([1, 3, 1])  
    with center_col:
        plot_closing_price_trend(nepse_combined_df, selected_symbols)


# 7. Correlation Matrix
with tabs[6]:
    # Correlation of Stock Price
    st.subheader("📊 Correlation Matrix")

    corr_matrix, fig = correlation_stock_price(nepse_combined_df)

    # Display the heatmap figure
    left_col, center_col, right_col = st.columns([1, 3, 1])  
    with center_col:
        st.pyplot(fig)

    # Optionally, display the correlation matrix as a table
    st.dataframe(corr_matrix)


# 8. Candlestick Chart
with tabs[7]:
    st.subheader("📈 Individual Stock Candlestick Chart")

    selected_symbol = st.selectbox("Selected Stock Symbol", options=sorted(nepse_combined_df['Symbol'].unique()), index=0)

    fig = plot_candlestick(nepse_combined_df, selected_symbol)
    if fig:
        st.plotly_chart(fig)
    else:
        st.write(f"No data available for symbol {selected_symbol}.")


# Display the stock future trend
with tabs[8]:
    st.subheader("🔮 Stock Trend Prediction (Visual + Heuristic)")

    symbol = st.selectbox("Select symbol", nepse_combined_df['Symbol'].unique())

    stock_data = nepse_combined_df[nepse_combined_df['Symbol'] == symbol].sort_values('Date')

    trend_signal = detect_trend(stock_data)
    trend_slope_text, slope_value = get_price_trend_slope(stock_data)

    st.markdown(f"**Current Trend Based on 120-Day MA:** {trend_signal}")
    st.markdown(f"**Recent Price Slope (30 days):** {trend_slope_text} (slope: `{slope_value:.4f}`)")


    # Project Future Prices

    st.subheader("📈 Next 10-Day Price Projection (Linear Trend)")

    symbol = st.selectbox("Select stock symbol", nepse_combined_df['Symbol'].unique())
    stock_data = nepse_combined_df[nepse_combined_df['Symbol'] == symbol]

    fig, future_dates, y_pred = plot_future_projection(stock_data, symbol)

    st.plotly_chart(fig)

    proj_df = pd.DataFrame({
        "Date": future_dates,
        "Projected Price": y_pred
    })
    st.dataframe(proj_df.style.format({"Projected Price": "{:.2f}"}))
import plotly.express as px
from symbol_to_group import symbol_to_group

def plot_stock_heatmap(df, date_filter=None):
    # Use only latest date if no filter given
    if date_filter is None:
        date_filter = df['Date'].max()
    
    # Filter to selected date only
    df = df[df['Date'] == date_filter]

    # Drop duplicates and keep latest info per symbol
    df = df.sort_values('Date').drop_duplicates('Symbol', keep='last')

    # Map sectors
    df['Group'] = df['Symbol'].map(symbol_to_group)

    # Drop rows without sector or DiffPercent
    df = df.dropna(subset=['Group', 'DiffPercent'])

    # Cap extreme % changes for better visuals
    df['CappedChange'] = df['DiffPercent'].clip(-5, 5)

    # Create treemap
    fig = px.treemap(
        df,
        path=['Group', 'Symbol'],
        values=[1] * len(df),  # Equal box size; change to Turnover if needed
        color='CappedChange',
        color_continuous_scale=['red', 'white', 'green'],
        range_color=[-5, 5],
        title=f'📊 NEPSE Heatmap - {date_filter}'
    )

    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Change: %{color:.2f}%<extra></extra>'
    )

    return fig

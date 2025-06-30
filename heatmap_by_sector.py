import plotly.express as px
from symbol_to_group import symbol_to_group

def plot_stock_heatmap(df, date_filter=None):
    if date_filter is None:
        date_filter = df['Date'].max()
    
    df = df[df['Date'] == date_filter]

    df = df.sort_values('Date').drop_duplicates('Symbol', keep='last')

    # Map sectors
    df['Group'] = df['Symbol'].map(symbol_to_group)

    df = df.dropna(subset=['Group', 'DiffPercent'])

    df['CappedChange'] = df['DiffPercent'].clip(-5, 5)

    fig = px.treemap(
        df,
        path=['Group', 'Symbol'],
        values=[1] * len(df),  
        color='CappedChange',
        color_continuous_scale=['red', 'white', 'green'],
        range_color=[-5, 5],
        title=f'📊 NEPSE Heatmap - {date_filter}'
    )

    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Change: %{color:.2f}%<extra></extra>'
    )

    return fig

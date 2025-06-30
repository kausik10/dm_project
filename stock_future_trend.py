def detect_trend(stock_data):
    
    latest_close = stock_data['ClosePrice'].iloc[-1]
    latest_ma = stock_data['120Days'].iloc[-1]

    if latest_close > latest_ma:
        return "📈 Bullish"
    elif latest_close < latest_ma:
        return "📉 Bearish"
    else:
        return "⚖️ Neutral"

from sklearn.linear_model import LinearRegression
import numpy as np

def get_price_trend_slope(stock_data, window=30):
    
    recent_data = stock_data[-window:]
    X = np.arange(window).reshape(-1, 1)
    y = recent_data['ClosePrice'].values.reshape(-1, 1)

    model = LinearRegression().fit(X, y)
    slope = model.coef_[0][0]

    if slope > 0:
        return "📈 Uptrend (Bullish)", slope
    elif slope < 0:
        return "📉 Downtrend (Bearish)", slope
    else:
        return "⚖️ Sideways", slope

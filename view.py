import plotly.graph_objects as go
from src.data import DataStore

df = DataStore(db_path="data/market_data.db").load_data("HK.00700", "DAY")
fig = go.Figure(go.Candlestick(x=df.index, open=df["open"], high=df["high"],low=df["low"], close=df["close"]))
fig.show()  # 浏览器打开交互图
import yfinance as yf

data = yf.download(["^GSPC", "^VIX"], start="2010-01-01", end="2024-01-01")['Close']

data.to_csv("data/data_market.csv")
print("File downloaded successfully")
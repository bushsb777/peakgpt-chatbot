import os
import requests

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
HEADERS = {"X-Finnhub-Token": FINNHUB_API_KEY}

# === Real-Time Prices ===
def get_crypto_price():
    symbols = ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT"]
    prices = {}
    for symbol in symbols:
        res = requests.get(f"https://finnhub.io/api/v1/quote?symbol={symbol}", headers=HEADERS)
        if res.ok:
            prices[symbol] = res.json()["c"]  # current price
    return prices

def get_stock_price():
    symbols = ["AAPL", "MSFT", "NVDA"]
    prices = {}
    for symbol in symbols:
        res = requests.get(f"https://finnhub.io/api/v1/quote?symbol={symbol}", headers=HEADERS)
        if res.ok:
            prices[symbol] = res.json()["c"]
    return prices

def get_etf_price():
    symbols = ["SPY", "QQQ", "VTI"]
    prices = {}
    for symbol in symbols:
        res = requests.get(f"https://finnhub.io/api/v1/quote?symbol={symbol}", headers=HEADERS)
        if res.ok:
            prices[symbol] = res.json()["c"]
    return prices

# === Sentiment (mocked for now or custom if you have access) ===
def get_crypto_sentiment():
    return "📉 Bitcoin slightly bearish, 📈 Ethereum neutral to bullish"

def get_stock_sentiment():
    return "📈 AAPL bullish, MSFT bullish, NVDA strong momentum"

def get_etf_sentiment():
    return "📉 SPY showing minor outflows, QQQ steady, VTI rising on large cap optimism"

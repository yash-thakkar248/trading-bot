import ccxt
import pandas as pd
import json
import yfinance as yf

def load_config():
    with open("configs/config.json", "r") as f:
        return json.load(f)

def fetch_ohlcv():
    config = load_config()
    exchange_id = config["exchange"]
    
    # Create exchange dynamically using ccxt.exchanges
    if exchange_id not in ccxt.exchanges:
        raise ValueError(f"Exchange {exchange_id} not supported by your CCXT version")
    
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()
    
    ohlcv = exchange.fetch_ohlcv(config["symbol"], config["timeframe"], limit=config["limit"])

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

from utils.logger import log_trade
import pandas as pd

def execute_trade(signal):
    """
    Execute trade and return formatted trade data for charting.
    FIXED: Now properly passes P&L and all data to charts.
    """
    if signal:
        # Log the trade
        log_trade(signal["action"], signal["price"], signal["reason"])
        
        # Format the trade for charting - include ALL signal data
        formatted_trade = {
            "action": signal["action"],
            "price": float(signal["price"]),
            "time": signal.get("time", pd.Timestamp.now()),
            "reason": signal["reason"]
        }
        
        # FIXED: Pass through P&L data if it exists (for SELL trades)
        if "pnl" in signal:
            formatted_trade["pnl"] = signal["pnl"]
        if "pnl_percent" in signal:
            formatted_trade["pnl_percent"] = signal["pnl_percent"]
        if "trade_amount" in signal:
            formatted_trade["trade_amount"] = signal["trade_amount"]
        if "shares" in signal:
            formatted_trade["shares"] = signal["shares"]
        
        # Convert time to pandas timestamp if it's not already
        if not isinstance(formatted_trade["time"], pd.Timestamp):
            formatted_trade["time"] = pd.to_datetime(formatted_trade["time"])
        
        return formatted_trade
    
    return None

def execute_portfolio_trade(signal, portfolio_balance=10000, position_size=0.1):
    """
    Execute trade with portfolio management (for future use).
    """
    if signal:
        trade_amount = portfolio_balance * position_size
        
        if signal["action"] == "BUY":
            shares = trade_amount / signal["price"]
            log_trade(signal["action"], signal["price"], 
                     f"{signal['reason']} | Amount: ${trade_amount:.2f} | Shares: {shares:.4f}")
        else:  # SELL
            log_trade(signal["action"], signal["price"], signal["reason"])
        
        # Return formatted trade
        return {
            "action": signal["action"],
            "price": float(signal["price"]),
            "time": signal.get("time", pd.Timestamp.now()),
            "reason": signal["reason"],
            "amount": trade_amount
        }
    
    return None
import logging

logging.basicConfig(
    filename="logs/trading.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_trade(action, price, reason):
    logging.info(f"TRADE: {action} at {price} because {reason}")

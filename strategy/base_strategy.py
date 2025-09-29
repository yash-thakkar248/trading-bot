import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

def load_config():
    with open("configs/config.json", "r") as f:
        return json.load(f)

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.trades = []
        self.total_pnl = 0.0
        self.position = None
        self.entry_price = None
        self.shares_held = 0.0
        self.invested_amount = 0.0
        
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Optional[dict]:
        """Generate trading signal based on market data"""
        pass
        
    @abstractmethod
    def get_strategy_params(self) -> dict:
        """Return strategy-specific parameters"""
        pass
    
    def reset(self):
        """Reset strategy state"""
        self.trades = []
        self.total_pnl = 0.0
        self.position = None
        self.entry_price = None
        self.shares_held = 0.0
        self.invested_amount = 0.0

class ImprovedTrendFollowingStrategy(BaseStrategy):
    """
    Improved Moving Average Strategy with better risk management
    """
    
    def __init__(self, config: dict):
        super().__init__("ImprovedTrendFollowing", config)
        strategy_config = config.get("strategies", {}).get("trend_following", {})
        
        # More aggressive settings for day trading
        self.fast_ma = strategy_config.get("fast_ma_period", 5)  # Faster
        self.slow_ma = strategy_config.get("slow_ma_period", 15)  # Faster
        
        # Add volume confirmation
        self.volume_period = strategy_config.get("volume_period", 10)
        self.volume_multiplier = strategy_config.get("volume_multiplier", 1.2)
        
        # Trend strength filter
        self.min_trend_strength = strategy_config.get("min_trend_strength", 0.5)  # 0.5% minimum price move
        
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()
    
    def check_trend_strength(self, df: pd.DataFrame) -> bool:
        """Check if trend has sufficient strength"""
        if len(df) < 10:
            return False
        
        recent_high = df['high'].tail(5).max()
        recent_low = df['low'].tail(5).min()
        price_range = ((recent_high - recent_low) / recent_low) * 100
        
        return price_range > self.min_trend_strength
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[dict]:
        if len(df) < max(self.fast_ma, self.slow_ma) + 5:
            return None
            
        current_price = df.iloc[-1]['close']
        current_time = df.iloc[-1].get('timestamp', '')
        
        # Calculate EMAs
        fast_ema = self.calculate_ema(df['close'], self.fast_ma)
        slow_ema = self.calculate_ema(df['close'], self.slow_ma)
        
        # Current and previous values
        current_fast = fast_ema.iloc[-1]
        current_slow = slow_ema.iloc[-1]
        prev_fast = fast_ema.iloc[-2] if len(fast_ema) >= 2 else current_fast
        prev_slow = slow_ema.iloc[-2] if len(slow_ema) >= 2 else current_slow
        
        # Volume confirmation
        avg_volume = df['volume'].tail(self.volume_period).mean()
        current_volume = df['volume'].iloc[-1]
        volume_confirm = current_volume > avg_volume * self.volume_multiplier
        
        # Check trend strength
        has_trend_strength = self.check_trend_strength(df)
        
        # Bullish crossover with confirmations
        if (prev_fast <= prev_slow and current_fast > current_slow and 
            self.position is None and volume_confirm and has_trend_strength):
            return {
                "action": "BUY", "price": current_price, "time": current_time,
                "reason": f"Improved Trend BUY: MA cross + Vol {current_volume/avg_volume:.1f}x + Strong trend",
                "signal_strength": 0.8
            }
        
        # Bearish crossover OR price falls below fast EMA (tighter exit)
        elif ((prev_fast >= prev_slow and current_fast < current_slow) or 
              (self.position == "LONG" and current_price < current_fast * 0.995)):
            if self.position == "LONG":
                return {
                    "action": "SELL", "price": current_price, "time": current_time,
                    "reason": f"Improved Trend SELL: MA cross or below fast EMA",
                    "signal_strength": 0.8
                }
        
        return None
    
    def get_strategy_params(self) -> dict:
        return {
            "type": "Improved MA Crossover",
            "fast_ma": self.fast_ma,
            "slow_ma": self.slow_ma,
            "volume_multiplier": self.volume_multiplier,
            "min_trend_strength": f"{self.min_trend_strength}%",
            "market_condition": "Trending with Volume"
        }

class ImprovedMeanReversionStrategy(BaseStrategy):
    """
    Improved Mean Reversion with multiple confirmations
    """
    
    def __init__(self, config: dict):
        super().__init__("ImprovedMeanReversion", config)
        strategy_config = config.get("strategies", {}).get("mean_reversion", {})
        
        # More sensitive settings
        self.rsi_period = strategy_config.get("rsi_period", 10)  # Faster RSI
        self.rsi_oversold = strategy_config.get("rsi_oversold", 25)  # More extreme
        self.rsi_overbought = strategy_config.get("rsi_overbought", 75)  # More extreme
        
        self.bb_period = strategy_config.get("bb_period", 15)  # Faster BB
        self.bb_std = strategy_config.get("bb_std", 2.5)  # Wider bands
        
        # Add stochastic for confirmation
        self.stoch_period = strategy_config.get("stoch_period", 7)
        
    def calculate_rsi(self, df: pd.DataFrame) -> Optional[float]:
        if len(df) < self.rsi_period + 1:
            return None
        
        closes = df['close']
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def calculate_bollinger_bands(self, df: pd.DataFrame):
        if len(df) < self.bb_period:
            return None, None, None
        
        closes = df['close'].tail(self.bb_period)
        sma = closes.mean()
        std = closes.std()
        upper = sma + (std * self.bb_std)
        lower = sma - (std * self.bb_std)
        return upper, sma, lower
    
    def calculate_stochastic(self, df: pd.DataFrame) -> Optional[float]:
        if len(df) < self.stoch_period:
            return None
        
        recent = df.tail(self.stoch_period)
        lowest_low = recent['low'].min()
        highest_high = recent['high'].max()
        current_close = df.iloc[-1]['close']
        
        k_percent = 100 * ((current_close - lowest_low) / (highest_high - lowest_low))
        return k_percent
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[dict]:
        if len(df) < max(self.rsi_period + 1, self.bb_period, self.stoch_period):
            return None
            
        current_price = df.iloc[-1]['close']
        current_time = df.iloc[-1].get('timestamp', '')
        
        rsi = self.calculate_rsi(df)
        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(df)
        stoch = self.calculate_stochastic(df)
        
        if None in [rsi, upper_bb, lower_bb, stoch]:
            return None
        
        # Multiple oversold confirmations
        if (rsi < self.rsi_oversold and 
            current_price <= lower_bb * 1.005 and  # Near or below lower BB
            stoch < 20 and  # Stochastic oversold
            self.position is None):
            return {
                "action": "BUY", "price": current_price, "time": current_time,
                "reason": f"Improved Mean Reversion BUY: RSI {rsi:.1f} + BB + Stoch {stoch:.1f}",
                "signal_strength": 0.85
            }
        
        # Multiple overbought confirmations
        elif (rsi > self.rsi_overbought and 
              current_price >= upper_bb * 0.995 and  # Near or above upper BB
              stoch > 80 and  # Stochastic overbought
              self.position == "LONG"):
            return {
                "action": "SELL", "price": current_price, "time": current_time,
                "reason": f"Improved Mean Reversion SELL: RSI {rsi:.1f} + BB + Stoch {stoch:.1f}",
                "signal_strength": 0.85
            }
        
        # Quick exit if price moves back to middle BB (take profits)
        elif (self.position == "LONG" and 
              current_price >= middle_bb and 
              rsi > 60):  # Some profit already made
            return {
                "action": "SELL", "price": current_price, "time": current_time,
                "reason": f"Mean Reversion profit take at BB middle",
                "signal_strength": 0.6
            }
        
        return None
    
    def get_strategy_params(self) -> dict:
        return {
            "type": "Improved RSI + BB + Stochastic",
            "rsi_period": self.rsi_period,
            "rsi_levels": f"{self.rsi_oversold}/{self.rsi_overbought}",
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "stoch_period": self.stoch_period,
            "market_condition": "Sideways/Ranging"
        }

# Import the day trading strategies from the previous artifact
class ScalpingStrategy:
    """1-Minute Scalping Strategy"""
    
    def __init__(self, config: dict):
        self.name = "Scalping_1m"
        self.config = config
        self.trades = []
        self.total_pnl = 0.0
        self.position = None
        self.entry_price = None
        self.shares_held = 0.0
        self.invested_amount = 0.0
        
        strategy_config = config.get("strategies", {}).get("scalping", {})
        self.fast_ema = strategy_config.get("fast_ema", 3)
        self.slow_ema = strategy_config.get("slow_ema", 8)
        self.stoch_k = strategy_config.get("stoch_k", 5)
        self.stoch_d = strategy_config.get("stoch_d", 3)
        self.volume_period = strategy_config.get("volume_period", 10)
        self.volume_multiplier = strategy_config.get("volume_multiplier", 1.2)
        self.stop_loss_pips = strategy_config.get("stop_loss_pips", 0.5)
        
    def reset(self):
        self.trades = []
        self.total_pnl = 0.0
        self.position = None
        self.entry_price = None
        self.shares_held = 0.0
        self.invested_amount = 0.0
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series):
        lowest_low = low.rolling(window=self.stoch_k).min()
        highest_high = high.rolling(window=self.stoch_k).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=self.stoch_d).mean()
        return k_percent, d_percent
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[dict]:
        if len(df) < 20:
            return None
        
        current_price = df.iloc[-1]['close']
        current_time = df.iloc[-1].get('timestamp', '')
        
        fast_ema = self.calculate_ema(df['close'], self.fast_ema)
        slow_ema = self.calculate_ema(df['close'], self.slow_ema)
        stoch_k, stoch_d = self.calculate_stochastic(df['high'], df['low'], df['close'])
        
        curr_fast = fast_ema.iloc[-1]
        curr_slow = slow_ema.iloc[-1]
        curr_stoch_k = stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50
        
        prev_fast = fast_ema.iloc[-2] if len(fast_ema) >= 2 else curr_fast
        prev_slow = slow_ema.iloc[-2] if len(slow_ema) >= 2 else curr_slow
        prev_stoch_k = stoch_k.iloc[-2] if len(stoch_k) >= 2 and not pd.isna(stoch_k.iloc[-2]) else curr_stoch_k
        
        # Volume confirmation
        avg_volume = df['volume'].tail(self.volume_period).mean()
        current_volume = df['volume'].iloc[-1]
        volume_confirm = current_volume > avg_volume * self.volume_multiplier
        
        # BUY Signal
        if (self.position is None and 
            prev_fast <= prev_slow and curr_fast > curr_slow and  # EMA crossover
            curr_stoch_k < 30 and curr_stoch_k > prev_stoch_k and  # Stoch rising from oversold
            volume_confirm):
            
            return {
                "action": "BUY", "price": current_price, "time": current_time,
                "reason": f"Scalp BUY: EMA cross + Stoch {curr_stoch_k:.1f} + Vol {current_volume/avg_volume:.1f}x",
                "signal_strength": 0.9
            }
        
        # SELL Signal
        elif (self.position == "LONG" and 
              (curr_stoch_k > 80 or 
               current_price <= self.entry_price * (1 - self.stop_loss_pips / 100))):
            
            reason = "Scalp SELL: "
            if curr_stoch_k > 80:
                reason += f"Stoch overbought ({curr_stoch_k:.1f})"
            else:
                reason += "Stop Loss"
            
            return {
                "action": "SELL", "price": current_price, "time": current_time,
                "reason": reason, "signal_strength": 0.9
            }
        
        return None
    
    def get_strategy_params(self) -> dict:
        return {
            "type": "1-Minute Scalping",
            "ema_periods": f"{self.fast_ema}/{self.slow_ema}",
            "stochastic": f"K:{self.stoch_k}, D:{self.stoch_d}",
            "stop_loss": f"{self.stop_loss_pips}%",
            "market_condition": "High Volume Trending"
        }
    

class SimpleStrategy(BaseStrategy):
    """Simple price comparison strategy"""
    
    def __init__(self, config: dict):
        super().__init__("Simple", config)
        
    def generate_signal(self, df: pd.DataFrame) -> Optional[dict]:
        if len(df) < 2:
            return None
            
        current_price = df.iloc[-1]['close']
        prev_price = df.iloc[-2]['close']
        current_time = df.iloc[-1].get('timestamp', '')
        
        if current_price > prev_price and self.position is None:
            return {"action": "BUY", "price": current_price, "time": current_time, 
                   "reason": "Price rising"}
        elif current_price < prev_price and self.position == "LONG":
            return {"action": "SELL", "price": current_price, "time": current_time,
                   "reason": "Price falling"}
        return None
    
    def get_strategy_params(self) -> dict:
        return {"type": "Simple Price Comparison", "market_condition": "Any"}

class MomentumBreakoutStrategy:
    """5-Minute Momentum Breakout Strategy"""
    
    def __init__(self, config: dict):
        self.name = "Momentum_5m"
        self.config = config
        self.trades = []
        self.total_pnl = 0.0
        self.position = None
        self.entry_price = None
        self.shares_held = 0.0
        self.invested_amount = 0.0
        
        strategy_config = config.get("strategies", {}).get("momentum_breakout", {})
        self.bb_period = strategy_config.get("bb_period", 20)
        self.bb_std = strategy_config.get("bb_std", 2.0)
        self.rsi_period = strategy_config.get("rsi_period", 7)
        self.rsi_momentum = strategy_config.get("rsi_momentum", 55)
        self.volume_period = strategy_config.get("volume_period", 20)
        self.volume_multiplier = strategy_config.get("volume_multiplier", 1.5)
        self.stop_loss_percent = strategy_config.get("stop_loss_percent", 1.0)
        
    def reset(self):
        self.trades = []
        self.total_pnl = 0.0
        self.position = None
        self.entry_price = None
        self.shares_held = 0.0
        self.invested_amount = 0.0
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices: pd.Series):
        sma = prices.rolling(window=self.bb_period).mean()
        std = prices.rolling(window=self.bb_period).std()
        upper_band = sma + (std * self.bb_std)
        lower_band = sma - (std * self.bb_std)
        return upper_band, sma, lower_band
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[dict]:
        if len(df) < max(self.bb_period, self.rsi_period) + 5:
            return None
        
        current_price = df.iloc[-1]['close']
        current_time = df.iloc[-1].get('timestamp', '')
        
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'])
        rsi = self.calculate_rsi(df['close'], self.rsi_period)
        
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        current_bb_upper = bb_upper.iloc[-1]
        current_bb_middle = bb_middle.iloc[-1]
        
        # Volume confirmation
        avg_volume = df['volume'].tail(self.volume_period).mean()
        current_volume = df['volume'].iloc[-1]
        volume_confirm = current_volume > avg_volume * self.volume_multiplier
        
        # BUY Signal: Breaking above upper BB
        if (self.position is None and 
            current_price > current_bb_upper * 1.001 and 
            current_rsi > self.rsi_momentum and 
            volume_confirm):
            
            return {
                "action": "BUY", "price": current_price, "time": current_time,
                "reason": f"Momentum BUY: BB breakout + RSI {current_rsi:.1f} + Vol {current_volume/avg_volume:.1f}x",
                "signal_strength": 0.85
            }
        
        # SELL Signal
        elif (self.position == "LONG" and 
              (current_price < current_bb_middle or 
               current_rsi < 40 or 
               current_price <= self.entry_price * (1 - self.stop_loss_percent / 100))):
            
            reason = "Momentum SELL: "
            if current_price <= self.entry_price * (1 - self.stop_loss_percent / 100):
                reason += "Stop Loss"
            elif current_rsi < 40:
                reason += f"Weak momentum (RSI {current_rsi:.1f})"
            else:
                reason += "Below BB middle"
            
            return {
                "action": "SELL", "price": current_price, "time": current_time,
                "reason": reason, "signal_strength": 0.85
            }
        
        return None
    
    def get_strategy_params(self) -> dict:
        return {
            "type": "5-Minute Momentum Breakout",
            "bollinger_bands": f"{self.bb_period}±{self.bb_std}σ",
            "rsi_period": self.rsi_period,
            "momentum_threshold": self.rsi_momentum,
            "stop_loss": f"{self.stop_loss_percent}%",
            "market_condition": "High Volatility Breakouts"
        }
from .bb_momentum_strategy import BBMomentumStrategy

class StrategyManager:
    """Manages multiple strategies with improved performance tracking"""
    
    def __init__(self):
        self.config = load_config()
        self.portfolio_config = self.config.get("portfolio", {})
        
        # Portfolio state
        self.initial_balance = self.portfolio_config.get("initial_balance", 10000)
        self.current_balance = self.initial_balance
        self.position_size_percent = self.portfolio_config.get("position_size_percent", 20)
        
        # Risk management
        self.stop_loss_percent = self.portfolio_config.get("stop_loss_percent")
        self.take_profit_percent = self.portfolio_config.get("take_profit_percent")
        
        # Initialize all available strategies
        self.available_strategies = {
            "simple": SimpleStrategy(self.config),
            "bb_momentum": BBMomentumStrategy(self.config)
        }
        
        # Load active strategies from config
        self.active_strategy_names = self.config.get("active_strategies", ["simple"])
        self.active_strategies = {
            name: self.available_strategies[name] 
            for name in self.active_strategy_names 
            if name in self.available_strategies
        }
        
        print(f"Loaded {len(self.active_strategies)} active strategies: {list(self.active_strategies.keys())}")
    
    def generate_signals(self, df: pd.DataFrame) -> List[dict]:
        """Generate signals from all active strategies"""
        signals = []
        
        for strategy_name, strategy in self.active_strategies.items():
            # Check stop loss/take profit for this strategy
            if strategy.position == "LONG" and strategy.entry_price:
                current_price = df.iloc[-1]['close']
                price_change = ((current_price - strategy.entry_price) / strategy.entry_price) * 100
                
                if self.stop_loss_percent and price_change <= -self.stop_loss_percent:
                    signal = self._create_sell_signal(strategy, current_price, df.iloc[-1].get('timestamp', ''),
                                                    f"Stop Loss ({price_change:+.2f}%)")
                    if signal:
                        signals.append(signal)
                        continue
                
                if self.take_profit_percent and price_change >= self.take_profit_percent:
                    signal = self._create_sell_signal(strategy, current_price, df.iloc[-1].get('timestamp', ''),
                                                    f"Take Profit ({price_change:+.2f}%)")
                    if signal:
                        signals.append(signal)
                        continue
            
            # Generate normal strategy signal
            raw_signal = strategy.generate_signal(df)
            if raw_signal:
                if raw_signal["action"] == "BUY" and strategy.position is None:
                    signal = self._create_buy_signal(strategy, raw_signal)
                elif raw_signal["action"] == "SELL" and strategy.position == "LONG":
                    signal = self._create_sell_signal(strategy, raw_signal["price"], 
                                                    raw_signal["time"], raw_signal["reason"])
                else:
                    continue
                    
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _create_buy_signal(self, strategy, raw_signal: dict) -> dict:
        """Create formatted BUY signal with position sizing"""
        current_price = raw_signal["price"]
        trade_amount = self.current_balance * (self.position_size_percent / 100)
        shares = trade_amount / current_price
        
        signal = {
            "strategy": strategy.name,
            "action": "BUY",
            "price": current_price,
            "time": raw_signal["time"],
            "reason": raw_signal["reason"],
            "trade_amount": trade_amount,
            "shares": shares,
            "portfolio_percent": self.position_size_percent,
            "signal_strength": raw_signal.get("signal_strength", 0.7)
        }
        
        # Update strategy state
        strategy.position = "LONG"
        strategy.entry_price = current_price
        strategy.shares_held = shares
        strategy.invested_amount = trade_amount
        strategy.trades.append(signal)
        
        return signal
    
    def _create_sell_signal(self, strategy, current_price: float, 
                           current_time: str, reason: str) -> dict:
        """Create formatted SELL signal with P&L calculation"""
        current_value = strategy.shares_held * current_price
        pnl = current_value - strategy.invested_amount
        pnl_percent = (pnl / strategy.invested_amount) * 100
        
        signal = {
            "strategy": strategy.name,
            "action": "SELL",
            "price": current_price,
            "time": current_time,
            "reason": f"{reason} (P&L: ${pnl:+.2f}, {pnl_percent:+.2f}%)",
            "trade_amount": current_value,
            "shares": strategy.shares_held,
            "pnl": pnl,
            "pnl_percent": pnl_percent
        }
        
        # Update strategy and portfolio state
        strategy.total_pnl += pnl
        strategy.trades.append(signal)
        self.current_balance += pnl
        
        # Reset strategy position
        strategy.position = None
        strategy.entry_price = None
        strategy.shares_held = 0.0
        strategy.invested_amount = 0.0
        
        return signal
    
    def get_summary(self) -> dict:
        """Get comprehensive trading summary"""
        all_trades = []
        total_pnl = 0.0
        strategy_summaries = {}
        
        for name, strategy in self.active_strategies.items():
            strategy_trades = len([t for t in strategy.trades if t['action'] == 'SELL'])
            win_trades = len([t for t in strategy.trades if t['action'] == 'SELL' and t.get('pnl', 0) > 0])
            win_rate = (win_trades / strategy_trades * 100) if strategy_trades > 0 else 0
            
            strategy_summaries[name] = {
                "trades": strategy_trades,
                "pnl": strategy.total_pnl,
                "win_rate": win_rate,
                "current_position": strategy.position,
                "params": strategy.get_strategy_params()
            }
            all_trades.extend(strategy.trades)
            total_pnl += strategy.total_pnl
        
        buy_signals = sum(1 for t in all_trades if t['action'] == 'BUY')
        sell_signals = sum(1 for t in all_trades if t['action'] == 'SELL')
        winning_trades = sum(1 for t in all_trades if t['action'] == 'SELL' and t.get('pnl', 0) > 0)
        overall_win_rate = (winning_trades / sell_signals * 100) if sell_signals > 0 else 0
        
        return {
            "total_trades": sell_signals,
            "total_pnl": total_pnl,
            "win_rate": overall_win_rate,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "all_trades": all_trades,
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "total_return_percent": ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            "active_strategies": list(self.active_strategies.keys()),
            "strategy_summaries": strategy_summaries
        }
    
    def reset(self):
        """Reset all strategies and portfolio state"""
        self.current_balance = self.initial_balance
        for strategy in self.active_strategies.values():
            strategy.reset()
    
    def get_config_summary(self):
        """Display current configuration"""
        print("OPTIMIZED DAY TRADING CONFIGURATION:")
        print(f"   Exchange: {self.config.get('exchange')}")
        print(f"   Symbol: {self.config.get('symbol')}")
        print(f"   Timeframe: {self.config.get('timeframe')}")
        print(f"   Portfolio: ${self.initial_balance:,.2f}")
        print(f"   Position Size: {self.position_size_percent}% per trade (smaller for day trading)")
        print(f"   Stop Loss: {self.stop_loss_percent}%")
        print(f"   Take Profit: {self.take_profit_percent}%")
        print(f"   Active Strategies: {', '.join(self.active_strategies.keys())}")
        
        for name, strategy in self.active_strategies.items():
            params = strategy.get_strategy_params()
            print(f"\n   {name.upper()} Strategy:")
            for key, value in params.items():
                print(f"     {key}: {value}")
        print()

# Global strategy manager instance
strategy_manager = StrategyManager()

# Compatibility functions for existing code
def simple_strategy(df):
    """Main entry point - generates signals from all active strategies"""
    signals = strategy_manager.generate_signals(df)
    return signals[0] if signals else None

def get_strategy_summary():
    return strategy_manager.get_summary()

def reset_strategy():
    strategy_manager.reset()

def get_config_summary():
    strategy_manager.get_config_summary()
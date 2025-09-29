import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

class ScalpingStrategy:
    """
    1-Minute Scalping Strategy
    Uses Stochastic + EMA + Volume for rapid entries/exits
    """
    
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
        
        # Fast EMAs for scalping
        self.fast_ema = strategy_config.get("fast_ema", 3)
        self.slow_ema = strategy_config.get("slow_ema", 8)
        
        # Stochastic settings for momentum
        self.stoch_k = strategy_config.get("stoch_k", 5)
        self.stoch_d = strategy_config.get("stoch_d", 3)
        self.stoch_oversold = strategy_config.get("stoch_oversold", 20)
        self.stoch_overbought = strategy_config.get("stoch_overbought", 80)
        
        # Volume confirmation
        self.volume_period = strategy_config.get("volume_period", 10)
        self.volume_multiplier = strategy_config.get("volume_multiplier", 1.2)
        
        # Risk management
        self.stop_loss_pips = strategy_config.get("stop_loss_pips", 0.5)  # 0.5% stop loss
        self.take_profit_ratio = strategy_config.get("take_profit_ratio", 2.0)  # 2:1 R:R
        
    def reset(self):
        self.trades = []
        self.total_pnl = 0.0
        self.position = None
        self.entry_price = None
        self.shares_held = 0.0
        self.invested_amount = 0.0
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
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
        
        # Calculate indicators
        fast_ema = self.calculate_ema(df['close'], self.fast_ema)
        slow_ema = self.calculate_ema(df['close'], self.slow_ema)
        stoch_k, stoch_d = self.calculate_stochastic(df['high'], df['low'], df['close'])
        
        # Current values
        curr_fast = fast_ema.iloc[-1]
        curr_slow = slow_ema.iloc[-1]
        curr_stoch_k = stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50
        curr_stoch_d = stoch_d.iloc[-1] if not pd.isna(stoch_d.iloc[-1]) else 50
        
        # Previous values for crossover detection
        prev_fast = fast_ema.iloc[-2] if len(fast_ema) >= 2 else curr_fast
        prev_slow = slow_ema.iloc[-2] if len(slow_ema) >= 2 else curr_slow
        prev_stoch_k = stoch_k.iloc[-2] if len(stoch_k) >= 2 and not pd.isna(stoch_k.iloc[-2]) else curr_stoch_k
        
        # Volume confirmation
        avg_volume = df['volume'].tail(self.volume_period).mean()
        current_volume = df['volume'].iloc[-1]
        volume_confirm = current_volume > avg_volume * self.volume_multiplier
        
        # BUY Signal: EMA crossover + Stochastic oversold + Volume
        if (self.position is None and 
            prev_fast <= prev_slow and curr_fast > curr_slow and  # EMA bullish crossover
            curr_stoch_k < 30 and curr_stoch_k > prev_stoch_k and  # Stoch rising from oversold
            volume_confirm):  # Volume confirmation
            
            return {
                "action": "BUY",
                "price": current_price,
                "time": current_time,
                "reason": f"Scalp BUY: EMA cross + Stoch oversold ({curr_stoch_k:.1f}) + Vol {current_volume/avg_volume:.1f}x",
                "stop_loss": current_price * (1 - self.stop_loss_pips / 100),
                "take_profit": current_price * (1 + (self.stop_loss_pips * self.take_profit_ratio) / 100)
            }
        
        # SELL Signal: EMA bearish cross + Stochastic overbought
        elif (self.position == "LONG" and 
              (prev_fast >= prev_slow and curr_fast < curr_slow or  # EMA bearish crossover
               curr_stoch_k > 80 or  # Stochastic overbought
               current_price <= self.entry_price * (1 - self.stop_loss_pips / 100))):  # Stop loss
            
            reason = "Scalp SELL: "
            if curr_stoch_k > 80:
                reason += f"Stoch overbought ({curr_stoch_k:.1f})"
            elif current_price <= self.entry_price * (1 - self.stop_loss_pips / 100):
                reason += "Stop Loss"
            else:
                reason += "EMA bearish cross"
            
            return {
                "action": "SELL",
                "price": current_price,
                "time": current_time,
                "reason": reason
            }
        
        return None
    
    def get_strategy_params(self) -> dict:
        return {
            "type": "1-Minute Scalping",
            "ema_periods": f"{self.fast_ema}/{self.slow_ema}",
            "stochastic": f"K:{self.stoch_k}, D:{self.stoch_d}",
            "risk_reward": f"1:{self.take_profit_ratio}",
            "stop_loss": f"{self.stop_loss_pips}%",
            "market_condition": "High Volume Trending"
        }

class MomentumBreakoutStrategy:
    """
    5-Minute Momentum Breakout Strategy
    Uses Bollinger Bands + RSI + Volume for breakout detection
    """
    
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
        
        # Bollinger Bands for volatility
        self.bb_period = strategy_config.get("bb_period", 20)
        self.bb_std = strategy_config.get("bb_std", 2.0)
        
        # RSI for momentum
        self.rsi_period = strategy_config.get("rsi_period", 7)  # Faster RSI for day trading
        self.rsi_momentum = strategy_config.get("rsi_momentum", 55)  # Above 55 for bullish momentum
        
        # Volume settings
        self.volume_period = strategy_config.get("volume_period", 20)
        self.volume_multiplier = strategy_config.get("volume_multiplier", 1.5)
        
        # Risk management
        self.stop_loss_percent = strategy_config.get("stop_loss_percent", 1.0)
        self.take_profit_percent = strategy_config.get("take_profit_percent", 2.0)
        
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
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
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
        
        # Calculate indicators
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'])
        rsi = self.calculate_rsi(df['close'], self.rsi_period)
        
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        current_bb_upper = bb_upper.iloc[-1]
        current_bb_lower = bb_lower.iloc[-1]
        current_bb_middle = bb_middle.iloc[-1]
        
        # Volume confirmation
        avg_volume = df['volume'].tail(self.volume_period).mean()
        current_volume = df['volume'].iloc[-1]
        volume_confirm = current_volume > avg_volume * self.volume_multiplier
        
        # BUY Signal: Breaking above upper BB + Strong RSI + Volume
        if (self.position is None and 
            current_price > current_bb_upper * 1.001 and  # Breaking above upper BB
            current_rsi > self.rsi_momentum and  # Strong momentum
            volume_confirm):  # Volume confirmation
            
            return {
                "action": "BUY",
                "price": current_price,
                "time": current_time,
                "reason": f"Momentum BUY: BB breakout + RSI {current_rsi:.1f} + Vol {current_volume/avg_volume:.1f}x",
                "stop_loss": current_price * (1 - self.stop_loss_percent / 100),
                "take_profit": current_price * (1 + self.take_profit_percent / 100)
            }
        
        # SELL Signal: Price below middle BB or stop loss
        elif (self.position == "LONG" and 
              (current_price < current_bb_middle or  # Below middle BB
               current_rsi < 40 or  # Weak momentum
               current_price <= self.entry_price * (1 - self.stop_loss_percent / 100))):  # Stop loss
            
            reason = "Momentum SELL: "
            if current_price <= self.entry_price * (1 - self.stop_loss_percent / 100):
                reason += "Stop Loss"
            elif current_rsi < 40:
                reason += f"Weak momentum (RSI {current_rsi:.1f})"
            else:
                reason += "Below BB middle"
            
            return {
                "action": "SELL",
                "price": current_price,
                "time": current_time,
                "reason": reason
            }
        
        return None
    
    def get_strategy_params(self) -> dict:
        return {
            "type": "5-Minute Momentum Breakout",
            "bollinger_bands": f"{self.bb_period}±{self.bb_std}σ",
            "rsi_period": self.rsi_period,
            "momentum_threshold": self.rsi_momentum,
            "stop_loss": f"{self.stop_loss_percent}%",
            "take_profit": f"{self.take_profit_percent}%",
            "market_condition": "High Volatility Breakouts"
        }

class TrendFollowingDayStrategy:
    """
    1-Hour Trend Following Strategy
    Uses multiple EMAs + ADX + Volume for strong trend detection
    """
    
    def __init__(self, config: dict):
        self.name = "TrendFollowing_1h"
        self.config = config
        self.trades = []
        self.total_pnl = 0.0
        self.position = None
        self.entry_price = None
        self.shares_held = 0.0
        self.invested_amount = 0.0
        
        strategy_config = config.get("strategies", {}).get("trend_following_day", {})
        
        # EMA settings for trend
        self.fast_ema = strategy_config.get("fast_ema", 8)
        self.medium_ema = strategy_config.get("medium_ema", 21)
        self.slow_ema = strategy_config.get("slow_ema", 55)
        
        # RSI for entries
        self.rsi_period = strategy_config.get("rsi_period", 14)
        self.rsi_oversold = strategy_config.get("rsi_oversold", 35)
        self.rsi_overbought = strategy_config.get("rsi_overbought", 65)
        
        # Volume
        self.volume_period = strategy_config.get("volume_period", 20)
        self.volume_multiplier = strategy_config.get("volume_multiplier", 1.3)
        
        # Risk management
        self.stop_loss_percent = strategy_config.get("stop_loss_percent", 2.0)
        self.take_profit_percent = strategy_config.get("take_profit_percent", 4.0)
        
    def reset(self):
        self.trades = []
        self.total_pnl = 0.0
        self.position = None
        self.entry_price = None
        self.shares_held = 0.0
        self.invested_amount = 0.0
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def identify_trend(self, df: pd.DataFrame) -> str:
        """Identify trend direction using EMA alignment"""
        if len(df) < self.slow_ema:
            return "UNKNOWN"
        
        fast_ema = self.calculate_ema(df['close'], self.fast_ema)
        medium_ema = self.calculate_ema(df['close'], self.medium_ema)
        slow_ema = self.calculate_ema(df['close'], self.slow_ema)
        
        current_fast = fast_ema.iloc[-1]
        current_medium = medium_ema.iloc[-1]
        current_slow = slow_ema.iloc[-1]
        
        if current_fast > current_medium > current_slow:
            return "UPTREND"
        elif current_fast < current_medium < current_slow:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[dict]:
        if len(df) < self.slow_ema + 10:
            return None
        
        current_price = df.iloc[-1]['close']
        current_time = df.iloc[-1].get('timestamp', '')
        
        # Identify trend
        trend = self.identify_trend(df)
        if trend == "UNKNOWN":
            return None
        
        # Calculate indicators
        fast_ema = self.calculate_ema(df['close'], self.fast_ema)
        rsi = self.calculate_rsi(df['close'], self.rsi_period)
        
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        current_fast_ema = fast_ema.iloc[-1]
        
        # Volume confirmation
        avg_volume = df['volume'].tail(self.volume_period).mean()
        current_volume = df['volume'].iloc[-1]
        volume_confirm = current_volume > avg_volume * self.volume_multiplier
        
        # BUY Signal: Uptrend + Price above fast EMA + RSI pullback + Volume
        if (self.position is None and 
            trend == "UPTREND" and 
            current_price > current_fast_ema and 
            self.rsi_oversold < current_rsi < 60 and  # RSI pullback but not overbought
            volume_confirm):
            
            return {
                "action": "BUY",
                "price": current_price,
                "time": current_time,
                "reason": f"Trend BUY: {trend} + Above EMA + RSI {current_rsi:.1f} + Vol {current_volume/avg_volume:.1f}x",
                "stop_loss": current_price * (1 - self.stop_loss_percent / 100),
                "take_profit": current_price * (1 + self.take_profit_percent / 100)
            }
        
        # SELL Signal: Trend change or RSI overbought or stop loss
        elif (self.position == "LONG" and 
              (trend == "DOWNTREND" or 
               current_price < current_fast_ema * 0.995 or  # Below fast EMA
               current_rsi > 75 or  # Very overbought
               current_price <= self.entry_price * (1 - self.stop_loss_percent / 100))):  # Stop loss
            
            reason = "Trend SELL: "
            if current_price <= self.entry_price * (1 - self.stop_loss_percent / 100):
                reason += "Stop Loss"
            elif trend == "DOWNTREND":
                reason += "Trend reversal"
            elif current_rsi > 75:
                reason += f"Overbought (RSI {current_rsi:.1f})"
            else:
                reason += "Below fast EMA"
            
            return {
                "action": "SELL",
                "price": current_price,
                "time": current_time,
                "reason": reason
            }
        
        return None
    
    def get_strategy_params(self) -> dict:
        return {
            "type": "1-Hour Trend Following",
            "ema_periods": f"{self.fast_ema}/{self.medium_ema}/{self.slow_ema}",
            "rsi_range": f"{self.rsi_oversold}-{self.rsi_overbought}",
            "stop_loss": f"{self.stop_loss_percent}%",
            "take_profit": f"{self.take_profit_percent}%",
            "market_condition": "Strong Trending Markets"
        }
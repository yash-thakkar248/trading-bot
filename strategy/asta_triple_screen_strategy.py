import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
# Remove the circular import - BaseStrategy will be imported by the StrategyManager
# from strategy.base_strategy import BaseStrategy

class ASTATripleScreenStrategy:
    """
    Advanced ASTA Triple Screen Strategy Implementation
    Multi-timeframe analysis with Tide, Wave, and Ripple components
    """
    
    def __init__(self, config: dict):
        # Initialize without inheriting from BaseStrategy to avoid circular import
        self.name = "ASTA_TripleScreen"
        self.config = config
        self.trades = []
        self.total_pnl = 0.0
        self.position = None
        self.entry_price = None
        self.shares_held = 0.0
        self.invested_amount = 0.0
        
        strategy_config = config.get("strategies", {}).get("asta_triple_screen", {})
        
        # Timeframe settings
        self.primary_tf = strategy_config.get("primary_timeframe", "1h")
        self.secondary_tf = strategy_config.get("secondary_timeframe", "15m")
        self.entry_tf = strategy_config.get("entry_timeframe", "5m")
        
        # EMA Settings
        self.fast_ema = strategy_config.get("fast_ema", 5)
        self.medium_ema = strategy_config.get("medium_ema", 13)
        self.slow_ema = strategy_config.get("slow_ema", 26)
        
        # RSI Settings
        self.rsi_period = strategy_config.get("rsi_period", 14)
        self.rsi_overbought = strategy_config.get("rsi_overbought", 60)
        self.rsi_oversold = strategy_config.get("rsi_oversold", 40)
        
        # Stochastic Settings
        self.stoch_k = strategy_config.get("stoch_k", 14)
        self.stoch_d = strategy_config.get("stoch_d", 3)
        
        # Bollinger Bands Settings
        self.bb_period = strategy_config.get("bb_period", 20)
        self.bb_std = strategy_config.get("bb_std", 2)
        
        # Fibonacci Settings
        self.fib_threshold = strategy_config.get("fib_threshold", 61.8)
        
        # Volume Settings
        self.volume_period = strategy_config.get("volume_period", 20)
        self.volume_multiplier = strategy_config.get("volume_multiplier", 1.5)
        
        # Minimum conditions for signal strength
        self.min_buy_conditions = strategy_config.get("min_buy_conditions", 5)
        self.min_sell_conditions = strategy_config.get("min_sell_conditions", 5)
    
    def reset(self):
        """Reset strategy state"""
        self.trades = []
        self.total_pnl = 0.0
        self.position = None
        self.entry_price = None
        self.shares_held = 0.0
        self.invested_amount = 0.0
        
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def identify_trend_direction(self, df: pd.DataFrame) -> str:
        """
        TIDE Analysis - Identify primary trend direction
        """
        if len(df) < self.slow_ema:
            return "UNKNOWN"
        
        close_prices = df['close']
        fast_ema = self.calculate_ema(close_prices, self.fast_ema)
        medium_ema = self.calculate_ema(close_prices, self.medium_ema)
        slow_ema = self.calculate_ema(close_prices, self.slow_ema)
        
        current_price = close_prices.iloc[-1]
        current_fast = fast_ema.iloc[-1]
        current_medium = medium_ema.iloc[-1]
        current_slow = slow_ema.iloc[-1]
        
        # Check for uptrend (higher highs and higher lows)
        recent_highs = df['high'].tail(10)
        recent_lows = df['low'].tail(10)
        
        higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-5]
        higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-5]
        
        # EMA alignment for trend confirmation
        ema_uptrend = current_fast > current_medium > current_slow
        ema_downtrend = current_fast < current_medium < current_slow
        
        if (higher_highs and higher_lows) or ema_uptrend:
            return "UPTREND"
        elif (not higher_highs and not higher_lows) or ema_downtrend:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    def check_wave_retracement(self, df: pd.DataFrame, trend: str) -> bool:
        """
        WAVE Analysis - Check for retracement against main trend
        """
        if len(df) < 50:
            return False
        
        recent_data = df.tail(20)
        
        if trend == "UPTREND":
            # Look for pullback in uptrend
            recent_high = recent_data['high'].max()
            current_price = df['close'].iloc[-1]
            retracement = (recent_high - current_price) / recent_high * 100
            
            return 2 < retracement < 15  # 2-15% retracement is healthy
            
        elif trend == "DOWNTREND":
            # Look for bounce in downtrend
            recent_low = recent_data['low'].min()
            current_price = df['close'].iloc[-1]
            bounce = (current_price - recent_low) / recent_low * 100
            
            return 2 < bounce < 15  # 2-15% bounce is expected
            
        return False
    
    def analyze_chart_patterns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Analyze bullish/bearish chart patterns
        """
        patterns = {
            "double_bottom": False,
            "double_top": False,
            "higher_lows": False,
            "lower_highs": False,
            "support_break": False,
            "resistance_break": False
        }
        
        if len(df) < 50:
            return patterns
        
        recent_data = df.tail(30)
        highs = recent_data['high']
        lows = recent_data['low']
        
        # Check for higher lows (bullish)
        recent_lows = lows.tail(10)
        if len(recent_lows) >= 4:
            patterns["higher_lows"] = recent_lows.iloc[-1] > recent_lows.iloc[-3] > recent_lows.iloc[-5]
        
        # Check for lower highs (bearish)
        recent_highs = highs.tail(10)
        if len(recent_highs) >= 4:
            patterns["lower_highs"] = recent_highs.iloc[-1] < recent_highs.iloc[-3] < recent_highs.iloc[-5]
        
        # Simple support/resistance break detection
        support_level = lows.tail(20).min()
        resistance_level = highs.tail(20).max()
        current_price = df['close'].iloc[-1]
        
        patterns["support_break"] = current_price < support_level * 0.998
        patterns["resistance_break"] = current_price > resistance_level * 1.002
        
        return patterns
    
    def check_ripple_conditions(self, df: pd.DataFrame, action: str) -> Dict[str, bool]:
        """
        RIPPLE Analysis - Short-term entry conditions
        """
        conditions = {}
        
        if len(df) < max(self.rsi_period, self.bb_period, self.stoch_k):
            return {}
        
        close_prices = df['close']
        high_prices = df['high']
        low_prices = df['low']
        volume = df['volume']
        
        # EMA Crossover
        fast_ema = self.calculate_ema(close_prices, self.fast_ema)
        medium_ema = self.calculate_ema(close_prices, self.medium_ema)
        
        prev_fast = fast_ema.iloc[-2] if len(fast_ema) >= 2 else fast_ema.iloc[-1]
        prev_medium = medium_ema.iloc[-2] if len(medium_ema) >= 2 else medium_ema.iloc[-1]
        curr_fast = fast_ema.iloc[-1]
        curr_medium = medium_ema.iloc[-1]
        
        if action == "BUY":
            conditions["ema_crossover"] = (prev_fast <= prev_medium and curr_fast > curr_medium)
        else:
            conditions["ema_crossover"] = (prev_fast >= prev_medium and curr_fast < curr_medium)
        
        # RSI Conditions
        rsi = self.calculate_rsi(close_prices, self.rsi_period)
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        if action == "BUY":
            conditions["rsi_condition"] = current_rsi > self.rsi_oversold and current_rsi < 80
            conditions["rsi_crossover"] = current_rsi > self.rsi_overbought  # Best when crossing above 60
        else:
            conditions["rsi_condition"] = current_rsi < self.rsi_overbought and current_rsi > 20
            conditions["rsi_crossover"] = current_rsi < self.rsi_oversold  # Best when crossing below 40
        
        # Stochastic Conditions
        stoch_k, stoch_d = self.calculate_stochastic(high_prices, low_prices, close_prices, 
                                                    self.stoch_k, self.stoch_d)
        
        if len(stoch_k) >= 2:
            current_k = stoch_k.iloc[-1]
            current_d = stoch_d.iloc[-1]
            
            if action == "BUY":
                conditions["stochastic"] = current_k > 20 and current_k < 80  # Not oversold
            else:
                conditions["stochastic"] = current_k < 80 and current_k > 20  # Not overbought
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close_prices, 
                                                                      self.bb_period, self.bb_std)
        current_price = close_prices.iloc[-1]
        
        if not pd.isna(bb_upper.iloc[-1]):
            if action == "BUY":
                conditions["bollinger"] = current_price <= bb_lower.iloc[-1] * 1.01  # Near lower band
            else:
                conditions["bollinger"] = current_price >= bb_upper.iloc[-1] * 0.99  # Near upper band
        
        # Volume Confirmation
        avg_volume = volume.tail(self.volume_period).mean()
        current_volume = volume.iloc[-1]
        conditions["volume"] = current_volume > avg_volume * self.volume_multiplier
        
        # Fibonacci Retracement (simplified)
        recent_high = high_prices.tail(20).max()
        recent_low = low_prices.tail(20).min()
        fib_level = recent_low + (recent_high - recent_low) * (self.fib_threshold / 100)
        
        if action == "BUY":
            conditions["fibonacci"] = current_price <= fib_level * 1.02
        else:
            conditions["fibonacci"] = current_price >= fib_level * 0.98
        
        return conditions
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[dict]:
        """
        Main signal generation using ASTA Triple Screen methodology
        """
        if len(df) < 50:  # Need sufficient data
            return None
        
        current_price = df.iloc[-1]['close']
        current_time = df.iloc[-1].get('timestamp', '')
        
        # Step 1: TIDE - Determine primary trend
        trend = self.identify_trend_direction(df)
        
        if trend == "UNKNOWN":
            return None
        
        # Step 2: WAVE - Check for retracement
        wave_retracement = self.check_wave_retracement(df, trend)
        
        # Step 3: Chart Patterns Analysis
        patterns = self.analyze_chart_patterns(df)
        
        # Step 4: RIPPLE - Entry conditions
        if trend in ["UPTREND", "SIDEWAYS"] and self.position is None:
            # Check BUY conditions
            ripple_conditions = self.check_ripple_conditions(df, "BUY")
            
            buy_score = 0
            buy_reasons = []
            
            # Mandatory conditions
            if trend == "UPTREND":
                buy_score += 2
                buy_reasons.append(f"Uptrend confirmed")
            
            if wave_retracement:
                buy_score += 1
                buy_reasons.append("Healthy retracement")
            
            # Ripple conditions scoring
            for condition, met in ripple_conditions.items():
                if met:
                    buy_score += 1
                    buy_reasons.append(f"{condition}")
            
            # Pattern bonuses
            if patterns.get("higher_lows"):
                buy_score += 1
                buy_reasons.append("Higher lows pattern")
            
            if patterns.get("resistance_break"):
                buy_score += 1
                buy_reasons.append("Resistance breakout")
            
            # Generate BUY signal if enough conditions met
            if buy_score >= self.min_buy_conditions:
                return {
                    "action": "BUY",
                    "price": current_price,
                    "time": current_time,
                    "reason": f"ASTA Triple Screen BUY (Score: {buy_score}) - {', '.join(buy_reasons[:3])}",
                    "signal_strength": min(buy_score / 10, 1.0),
                    "conditions_met": buy_score
                }
        
        elif trend in ["DOWNTREND", "SIDEWAYS"] and self.position == "LONG":
            # Check SELL conditions
            ripple_conditions = self.check_ripple_conditions(df, "SELL")
            
            sell_score = 0
            sell_reasons = []
            
            # Mandatory conditions
            if trend == "DOWNTREND":
                sell_score += 2
                sell_reasons.append("Downtrend confirmed")
            
            if wave_retracement:
                sell_score += 1
                sell_reasons.append("Retracement against trend")
            
            # Ripple conditions scoring
            for condition, met in ripple_conditions.items():
                if met:
                    sell_score += 1
                    sell_reasons.append(f"{condition}")
            
            # Pattern bonuses
            if patterns.get("lower_highs"):
                sell_score += 1
                sell_reasons.append("Lower highs pattern")
            
            if patterns.get("support_break"):
                sell_score += 1
                sell_reasons.append("Support breakdown")
            
            # Generate SELL signal if enough conditions met
            if sell_score >= self.min_sell_conditions:
                return {
                    "action": "SELL",
                    "price": current_price,
                    "time": current_time,
                    "reason": f"ASTA Triple Screen SELL (Score: {sell_score}) - {', '.join(sell_reasons[:3])}",
                    "signal_strength": min(sell_score / 10, 1.0),
                    "conditions_met": sell_score
                }
        
        return None
    
    def get_strategy_params(self) -> dict:
        """Return strategy parameters for reporting"""
        return {
            "type": "ASTA Triple Screen",
            "timeframes": f"{self.primary_tf}/{self.secondary_tf}/{self.entry_tf}",
            "ema_periods": f"{self.fast_ema}/{self.medium_ema}/{self.slow_ema}",
            "rsi_levels": f"{self.rsi_oversold}/{self.rsi_overbought}",
            "min_conditions": f"Buy:{self.min_buy_conditions}, Sell:{self.min_sell_conditions}",
            "market_condition": "Multi-timeframe Analysis"
        }
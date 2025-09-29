import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

class BBMomentumStrategy:
    """
    Bollinger Band Challenge with Trendline Break Strategy
    Multi-timeframe momentum strategy based on the provided checklist
    """
    
    def __init__(self, config: dict):
        self.name = "BB_Momentum"
        self.config = config
        strategy_config = config.get("strategies", {}).get("bb_momentum", {})
        
        # Strategy state
        self.position = None
        self.entry_price = None
        self.trades = []
        self.total_pnl = 0.0
        self.shares_held = 0.0
        self.invested_amount = 0.0
        
        # Bollinger Bands settings
        self.bb_period = strategy_config.get("bb_period", 20)
        self.bb_std = strategy_config.get("bb_std", 2)
        
        # EMA settings
        self.fast_ema = strategy_config.get("fast_ema", 5)
        self.medium_ema = strategy_config.get("medium_ema", 13)
        self.slow_ema = strategy_config.get("slow_ema", 26)
        self.price_ema = strategy_config.get("price_ema", 50)
        
        # RSI settings
        self.rsi_period = strategy_config.get("rsi_period", 14)
        self.rsi_bullish = strategy_config.get("rsi_bullish", 50)
        self.rsi_bearish = strategy_config.get("rsi_bearish", 50)
        
        # ADX settings
        self.adx_period = strategy_config.get("adx_period", 14)
        self.adx_threshold = strategy_config.get("adx_threshold", 15)
        
        # Volume settings
        self.volume_period = strategy_config.get("volume_period", 20)
        
        # Minimum conditions for signal
        self.min_buy_conditions = strategy_config.get("min_buy_conditions", 4)
        self.min_sell_conditions = strategy_config.get("min_sell_conditions", 4)
        
    def reset(self):
        """Reset strategy state"""
        self.position = None
        self.entry_price = None
        self.trades = []
        self.total_pnl = 0.0
        self.shares_held = 0.0
        self.invested_amount = 0.0
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
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
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movements
        dm_pos = high.diff()
        dm_neg = -low.diff()
        
        dm_pos = dm_pos.where((dm_pos > dm_neg) & (dm_pos > 0), 0)
        dm_neg = dm_neg.where((dm_neg > dm_pos) & (dm_neg > 0), 0)
        
        # Smoothed values
        tr_smooth = true_range.ewm(alpha=1/period, adjust=False).mean()
        dm_pos_smooth = dm_pos.ewm(alpha=1/period, adjust=False).mean()
        dm_neg_smooth = dm_neg.ewm(alpha=1/period, adjust=False).mean()
        
        # Directional Indicators
        di_pos = 100 * dm_pos_smooth / tr_smooth
        di_neg = 100 * dm_neg_smooth / tr_smooth
        
        # ADX
        dx = 100 * abs(di_pos - di_neg) / (di_pos + di_neg)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx
    
    def detect_trendline_break(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Detect trendline breaks and trend direction"""
        if len(df) < 20:
            return {"uptick": False, "downtick": False}
        
        recent_data = df.tail(10)
        
        # Simple trend detection based on recent price action
        recent_highs = recent_data['high']
        recent_lows = recent_data['low']
        
        # Check for uptick (series of higher lows or breaking above recent resistance)
        uptick = (recent_lows.iloc[-1] > recent_lows.iloc[-3]) and (recent_lows.iloc[-3] > recent_lows.iloc[-5])
        
        # Check for downtick (series of lower highs or breaking below recent support)
        downtick = (recent_highs.iloc[-1] < recent_highs.iloc[-3]) and (recent_highs.iloc[-3] < recent_highs.iloc[-5])
        
        return {"uptick": uptick, "downtick": downtick}
    
    def check_bb_position(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check Bollinger Band position conditions"""
        if len(df) < self.bb_period:
            return {}
        
        close_prices = df['close']
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close_prices, self.bb_period, self.bb_std)
        
        current_price = close_prices.iloc[-1]
        current_upper = bb_upper.iloc[-1]
        current_lower = bb_lower.iloc[-1]
        current_middle = bb_middle.iloc[-1]
        
        conditions = {}
        
        # BBUC (Bollinger Band Upper Challenge) - price in upper half
        conditions["bbuc"] = current_price > current_middle
        
        # BBDC (Bollinger Band Down Challenge) - price in lower half  
        conditions["bbdc"] = current_price < current_middle
        
        # TLBO (Trendline Breakout) conditions
        conditions["tlbo"] = current_price > current_upper * 0.998  # Near or above upper band
        
        # TLBD (Trendline Breakdown) conditions
        conditions["tlbd"] = current_price < current_lower * 1.002  # Near or below lower band
        
        return conditions
    
    def check_ema_conditions(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check EMA-related conditions"""
        if len(df) < max(self.fast_ema, self.medium_ema, self.slow_ema, self.price_ema):
            return {}
        
        close_prices = df['close']
        fast_ema = self.calculate_ema(close_prices, self.fast_ema)
        medium_ema = self.calculate_ema(close_prices, self.medium_ema)
        slow_ema = self.calculate_ema(close_prices, self.slow_ema)
        price_ema = self.calculate_ema(close_prices, self.price_ema)
        
        current_price = close_prices.iloc[-1]
        
        conditions = {}
        
        # 5 EMA crossover conditions (last 3 periods)
        if len(fast_ema) >= 3 and len(medium_ema) >= 3:
            # Positive crossover
            conditions["ema_positive_cross"] = (fast_ema.iloc[-1] > medium_ema.iloc[-1]) and \
                                             (fast_ema.iloc[-2] <= medium_ema.iloc[-2])
            
            # Negative crossover
            conditions["ema_negative_cross"] = (fast_ema.iloc[-1] < medium_ema.iloc[-1]) and \
                                             (fast_ema.iloc[-2] >= medium_ema.iloc[-2])
        
        # Price vs 50 EMA conditions
        conditions["price_above_50ema"] = current_price > price_ema.iloc[-1]
        conditions["price_below_50ema"] = current_price < price_ema.iloc[-1]
        
        return conditions
    
    def check_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check RSI and ADX momentum conditions"""
        if len(df) < max(self.rsi_period, self.adx_period):
            return {}
        
        close_prices = df['close']
        high_prices = df['high']
        low_prices = df['low']
        
        rsi = self.calculate_rsi(close_prices, self.rsi_period)
        adx = self.calculate_adx(high_prices, low_prices, close_prices, self.adx_period)
        
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        
        conditions = {}
        
        # RSI conditions
        conditions["rsi_bullish"] = current_rsi > self.rsi_bullish
        conditions["rsi_bearish"] = current_rsi < self.rsi_bearish
        conditions["rsi_strong_bullish"] = current_rsi > 60  # RSI crossing above 60
        conditions["rsi_strong_bearish"] = current_rsi < 40  # RSI crossing below 40
        
        # ADX conditions
        conditions["adx_strong"] = current_adx > self.adx_threshold
        
        return conditions
    
    def check_volume_conditions(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check volume conditions"""
        if len(df) < self.volume_period:
            return {}
        
        volume = df['volume']
        avg_volume = volume.tail(self.volume_period).mean()
        current_volume = volume.iloc[-1]
        
        conditions = {}
        conditions["above_avg_volume"] = current_volume > avg_volume
        
        return conditions
    
    def check_pattern_conditions(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check for higher lows/lower highs patterns"""
        if len(df) < 15:
            return {}
        
        recent_data = df.tail(10)
        conditions = {}
        
        # At least two higher lows (bullish pattern)
        lows = recent_data['low']
        if len(lows) >= 6:
            higher_lows = (lows.iloc[-1] > lows.iloc[-3]) and (lows.iloc[-3] > lows.iloc[-5])
            conditions["higher_lows"] = higher_lows
        
        # At least two lower highs (bearish pattern)
        highs = recent_data['high']
        if len(highs) >= 6:
            lower_highs = (highs.iloc[-1] < highs.iloc[-3]) and (highs.iloc[-3] < highs.iloc[-5])
            conditions["lower_highs"] = lower_highs
        
        return conditions
    
    def check_support_resistance(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check for major support/resistance levels"""
        if len(df) < 50:
            return {}
        
        recent_data = df.tail(30)
        current_price = df['close'].iloc[-1]
        
        # Find recent support and resistance levels
        resistance = recent_data['high'].max()
        support = recent_data['low'].min()
        
        # Check if no immediate major resistance (for buys)
        price_to_resistance = (resistance - current_price) / current_price
        no_major_resistance = price_to_resistance > 0.02  # At least 2% room to resistance
        
        # Check if no immediate major support (for sells)
        price_to_support = (current_price - support) / current_price
        no_major_support = price_to_support > 0.02  # At least 2% room to support
        
        return {
            "no_major_resistance": no_major_resistance,
            "no_major_support": no_major_support
        }
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[dict]:
        """Generate trading signals based on BB Momentum strategy"""
        if len(df) < 50:  # Need sufficient data
            return None
        
        current_price = df.iloc[-1]['close']
        current_time = df.iloc[-1].get('timestamp', '')
        
        # Get all condition checks
        trendline_conditions = self.detect_trendline_break(df)
        bb_conditions = self.check_bb_position(df)
        ema_conditions = self.check_ema_conditions(df)
        momentum_conditions = self.check_momentum_indicators(df)
        volume_conditions = self.check_volume_conditions(df)
        pattern_conditions = self.check_pattern_conditions(df)
        support_resistance = self.check_support_resistance(df)
        
        # BUY SIGNAL CONDITIONS
        if self.position is None:
            buy_score = 0
            buy_reasons = []
            
            # Mandatory Conditions (TIDE)
            mandatory_met = 0
            
            # BBUC OR Price in Upper Half
            if bb_conditions.get("bbuc", False):
                mandatory_met += 1
                buy_reasons.append("Price in upper BB half")
            
            # TI Uptick (better above Zero Line) 
            if trendline_conditions.get("uptick", False):
                mandatory_met += 1
                buy_reasons.append("Trendline uptick")
            
            # RSI > 50 (better when RSI crossing above 60)
            if momentum_conditions.get("rsi_bullish", False):
                mandatory_met += 1
                buy_reasons.append(f"RSI bullish")
                
            if momentum_conditions.get("rsi_strong_bullish", False):
                buy_score += 1  # Bonus for strong RSI
                buy_reasons.append("RSI >60")
            
            # Only proceed if mandatory conditions are met
            if mandatory_met >= 3:
                buy_score += mandatory_met
                
                # Additional BUY conditions
                if bb_conditions.get("tlbo", False):
                    buy_score += 1
                    buy_reasons.append("TLBO breakout")
                
                if volume_conditions.get("above_avg_volume", False):
                    buy_score += 1
                    buy_reasons.append("Above avg volume")
                
                if pattern_conditions.get("higher_lows", False):
                    buy_score += 1
                    buy_reasons.append("Higher lows pattern")
                
                if ema_conditions.get("ema_positive_cross", False):
                    buy_score += 1
                    buy_reasons.append("EMA positive cross")
                
                if momentum_conditions.get("adx_strong", False):
                    buy_score += 1
                    buy_reasons.append("Strong ADX")
                
                if ema_conditions.get("price_above_50ema", False):
                    buy_score += 1
                    buy_reasons.append("Price > 50 EMA")
                
                if support_resistance.get("no_major_resistance", False):
                    buy_score += 1
                    buy_reasons.append("No major resistance")
                
                # Generate BUY signal if enough conditions met
                if buy_score >= self.min_buy_conditions:
                    return {
                        "action": "BUY",
                        "price": current_price,
                        "time": current_time,
                        "reason": f"BB Momentum BUY (Score: {buy_score}) - {', '.join(buy_reasons[:3])}",
                        "signal_strength": min(buy_score / 8, 1.0),
                        "conditions_met": buy_score
                    }
        
        # SELL SIGNAL CONDITIONS
        elif self.position == "LONG":
            sell_score = 0
            sell_reasons = []
            
            # Mandatory Conditions (TIDE)
            mandatory_met = 0
            
            # BBDC OR Price in Lower Half
            if bb_conditions.get("bbdc", False):
                mandatory_met += 1
                sell_reasons.append("Price in lower BB half")
            
            # TI Downtick (better below Zero Line)
            if trendline_conditions.get("downtick", False):
                mandatory_met += 1
                sell_reasons.append("Trendline downtick")
            
            # RSI < 50 (better when RSI crossing below 40)
            if momentum_conditions.get("rsi_bearish", False):
                mandatory_met += 1
                sell_reasons.append("RSI bearish")
                
            if momentum_conditions.get("rsi_strong_bearish", False):
                sell_score += 1  # Bonus for strong RSI
                sell_reasons.append("RSI <40")
            
            # Only proceed if mandatory conditions are met
            if mandatory_met >= 3:
                sell_score += mandatory_met
                
                # Additional SELL conditions
                if bb_conditions.get("tlbd", False):
                    sell_score += 1
                    sell_reasons.append("TLBD breakdown")
                
                if volume_conditions.get("above_avg_volume", False):
                    sell_score += 1
                    sell_reasons.append("Above avg volume")
                
                if pattern_conditions.get("lower_highs", False):
                    sell_score += 1
                    sell_reasons.append("Lower highs pattern")
                
                if ema_conditions.get("ema_negative_cross", False):
                    sell_score += 1
                    sell_reasons.append("EMA negative cross")
                
                if momentum_conditions.get("adx_strong", False):
                    sell_score += 1
                    sell_reasons.append("Strong ADX")
                
                if ema_conditions.get("price_below_50ema", False):
                    sell_score += 1
                    sell_reasons.append("Price < 50 EMA")
                
                if support_resistance.get("no_major_support", False):
                    sell_score += 1
                    sell_reasons.append("No major support")
                
                # Generate SELL signal if enough conditions met
                if sell_score >= self.min_sell_conditions:
                    return {
                        "action": "SELL",
                        "price": current_price,
                        "time": current_time,
                        "reason": f"BB Momentum SELL (Score: {sell_score}) - {', '.join(sell_reasons[:3])}",
                        "signal_strength": min(sell_score / 8, 1.0),
                        "conditions_met": sell_score
                    }
        
        return None
    
    def get_strategy_params(self) -> dict:
        """Return strategy parameters for reporting"""
        return {
            "type": "Bollinger Band Challenge with Trendline Break",
            "bb_period": self.bb_period,
            "ema_periods": f"{self.fast_ema}/{self.medium_ema}/{self.slow_ema}",
            "rsi_levels": f"{self.rsi_bearish}/{self.rsi_bullish}",
            "min_conditions": f"Buy:{self.min_buy_conditions}, Sell:{self.min_sell_conditions}",
            "market_condition": "Momentum/Trending Markets"
        }
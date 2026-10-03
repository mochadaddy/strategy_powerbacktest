"""
Market Regime Strategy

Classifies every bar into one of three regimes and trades accordingly:
    - Uptrend   : hold the position (buy if flat and enter_on_uptrend is enabled)
    - Range     : mean reversion, buy near the lower band / sell near the upper band
    - Downtrend : liquidate the position

Regime detection:
    - Uptrend   : ADX > adx_threshold, +DI > -DI, close > MA and MA rising
    - Downtrend : -DI > +DI, close < MA, MA falling, and either ADX > adx_threshold
                  or the MA fell by at least down_slope_pct over ma_slope_period bars
                  (catches slow declines that ADX misses)
    - Range     : everything else

Range trading (mean reversion):
    - Buy  (1) : close <= lower Bollinger band or RSI < rsi_oversold
    - Sell (-1): close >= upper Bollinger band or RSI > rsi_overbought

Parameters:
    - adx_period (int): ADX / DI period (default: 14)
    - adx_threshold (float): ADX level separating trend from range (default: 25)
    - ma_period (int): Trend moving average period (default: 50)
    - ma_slope_period (int): Bars used to measure the MA slope (default: 10)
    - down_slope_pct (float): MA drop that marks a downtrend even with low ADX (default: 0.02)
    - bb_period (int): Bollinger band period (default: 20)
    - bb_std (float): Bollinger band width in standard deviations (default: 2.0)
    - rsi_period (int): RSI period (default: 14)
    - rsi_oversold (float): RSI buy level in a range (default: 30)
    - rsi_overbought (float): RSI sell level in a range (default: 70)
    - enter_on_uptrend (bool): Buy when flat in an uptrend (default: True)

Example:
    strategy = RegimeStrategy({"adx_threshold": 25, "ma_period": 50})
    signals = strategy.generate_signals(strategy.calculate_indicators(price_data))
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import talib

from .base_strategy import BaseStrategy

UPTREND = "up"
RANGE = "range"
DOWNTREND = "down"


class RegimeStrategy(BaseStrategy):
    def __init__(self, parameters: Dict[str, Any] = None):
        parameters = parameters or {}
        self.adx_period = parameters.get("adx_period", 14)
        self.adx_threshold = parameters.get("adx_threshold", 25)
        self.ma_period = parameters.get("ma_period", 50)
        self.ma_slope_period = parameters.get("ma_slope_period", 10)
        self.down_slope_pct = parameters.get("down_slope_pct", 0.02)
        self.bb_period = parameters.get("bb_period", 20)
        self.bb_std = parameters.get("bb_std", 2.0)
        self.rsi_period = parameters.get("rsi_period", 14)
        self.rsi_oversold = parameters.get("rsi_oversold", 30)
        self.rsi_overbought = parameters.get("rsi_overbought", 70)
        self.enter_on_uptrend = parameters.get("enter_on_uptrend", True)
        self.validate_parameters()
        super().__init__(parameters)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add ADX, +DI, -DI, MA, MA_Slope, BB_Upper, BB_Middle, BB_Lower, RSI
        and Regime columns. Requires high, low and close columns.
        """
        data = data.copy()
        high = data["high"].astype(float).values
        low = data["low"].astype(float).values
        close = data["close"].astype(float).values

        data["ADX"] = talib.ADX(high, low, close, timeperiod=self.adx_period)
        data["Plus_DI"] = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        data["Minus_DI"] = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)
        data["MA"] = talib.SMA(close, timeperiod=self.ma_period)
        data["MA_Slope"] = data["MA"] - data["MA"].shift(self.ma_slope_period)
        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
        )
        data["BB_Upper"] = upper
        data["BB_Middle"] = middle
        data["BB_Lower"] = lower
        data["RSI"] = talib.RSI(close, timeperiod=self.rsi_period)
        data["Regime"] = self._classify_regime(data)
        return data

    def _classify_regime(self, data: pd.DataFrame) -> pd.Series:
        trending = data["ADX"] > self.adx_threshold
        uptrend = (
            trending
            & (data["Plus_DI"] > data["Minus_DI"])
            & (data["close"] > data["MA"])
            & (data["MA_Slope"] > 0)
        )
        steep_decline = data["MA_Slope"] <= -self.down_slope_pct * data["MA"].shift(
            self.ma_slope_period
        )
        downtrend = (
            (trending | steep_decline)
            & (data["Minus_DI"] > data["Plus_DI"])
            & (data["close"] < data["MA"])
            & (data["MA_Slope"] < 0)
        )
        return pd.Series(
            np.select([uptrend, downtrend], [UPTREND, DOWNTREND], default=RANGE),
            index=data.index,
        )

    def get_required_warmup_period(self) -> int:
        """ADX needs about twice its period to stabilise"""
        return max(
            2 * self.adx_period,
            self.ma_period + self.ma_slope_period,
            self.bb_period,
            self.rsi_period + 1,
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        1 (Buy), -1 (Sell), 0 (Hold). The engine only buys when flat and only
        sells when holding, so 1 in an uptrend means "hold or enter".
        """
        signals = pd.Series(0, index=data.index)
        regime = data["Regime"]
        close = data["close"]

        if self.enter_on_uptrend:
            signals[regime == UPTREND] = 1

        in_range = regime == RANGE
        buy_low = (close <= data["BB_Lower"]) | (data["RSI"] < self.rsi_oversold)
        sell_high = (close >= data["BB_Upper"]) | (data["RSI"] > self.rsi_overbought)
        signals[in_range & buy_low] = 1
        signals[in_range & sell_high & ~buy_low] = -1

        signals[regime == DOWNTREND] = -1

        ready = data[["ADX", "MA_Slope", "BB_Lower", "RSI"]].notna().all(axis=1)
        signals[~ready] = 0
        return signals

    def validate_parameters(self) -> bool:
        periods = {
            "adx_period": self.adx_period,
            "ma_period": self.ma_period,
            "ma_slope_period": self.ma_slope_period,
            "bb_period": self.bb_period,
            "rsi_period": self.rsi_period,
        }
        for name, value in periods.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0")
        if self.down_slope_pct <= 0:
            raise ValueError("down_slope_pct must be greater than 0")
        if self.bb_std <= 0:
            raise ValueError("bb_std must be greater than 0")
        if not 0 < self.adx_threshold < 100:
            raise ValueError("adx_threshold must be between 0 and 100")
        if not 0 < self.rsi_oversold < self.rsi_overbought < 100:
            raise ValueError("Require 0 < rsi_oversold < rsi_overbought < 100")
        return True

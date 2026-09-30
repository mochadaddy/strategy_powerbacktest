"""
Moving Average Cross Strategy

This strategy implements a classic technical analysis approach using two moving averages
to generate trading signals. It identifies potential trend changes by monitoring the
crossover points between a short-term and a long-term moving average.

Key Components:
    - Short-term Moving Average: More responsive to recent price changes
    - Long-term Moving Average: Establishes the broader trend
    
Trading Logic:
    - Buy (1): When short-term MA crosses above long- term MA (bullish signal)
    - Sell (-1): When short-term MA crosses below long-term MA (bearish signal)
    - Hold (0): When no crossing occurs

Parameters:
    - short_window (int): Period for short-term moving average (default: 20)
    - long_window (int): Period for long-term moving average (default: 50)

Example:
    strategy = MovingAverageCrossStrategy({
        'short_window': 20,
        'long_window': 50
    })
    signals = strategy.generate_signals(price_data)

Notes:
    - Best suited for trending markets
    - May generate false signals in sideways/choppy markets
    - Consider using with additional filters for better accuracy
    - Performance depends heavily on the chosen MA periods
"""

from typing import Dict, Any
import pandas as pd
from .base_strategy import BaseStrategy


class MovingAverageCrossStrategy(BaseStrategy):
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the Moving Average Cross Strategy.

        Args:
            parameters (Dict[str, Any], optional): Strategy parameters including:
                - short_window: Period for short-term moving average
                - long_window: Period for long-term moving average
                If None, defaults will be used.

        Raises:
            ValueError: If short_window >= long_window
        """
        parameters = parameters or {}  # Ensure parameters is at least an empty dict
        self.short_window = parameters.get("short_window", 20)
        self.long_window = parameters.get("long_window", 50)
        self.validate_parameters()
        super().__init__(parameters)

        # Validate parameters
        if self.short_window >= self.long_window:
            raise ValueError(
                f"Short window ({self.short_window}) must be less than "
                f"long window ({self.long_window})"
            )

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators used by the strategy.

        This method computes both short-term and long-term simple moving averages
        of the closing prices.

        Args:
            data (pd.DataFrame): Historical price data with required columns:
                - close: Closing prices

        Returns:
            pd.DataFrame: Original data with additional columns:
                - SMA_short: Short-term simple moving average
                - SMA_long: Long-term simple moving average
        """
        data = data.copy()

        # Calculate short-term moving average
        data["SMA_short"] = (
            data["close"]
            .rolling(
                window=self.short_window,
                min_periods=1,  # Allow partial windows at the start
            )
            .mean()
        )

        # Calculate long-term moving average
        data["SMA_long"] = (
            data["close"]
            .rolling(
                window=self.long_window,
                min_periods=1,  # Allow partial windows at the start
            )
            .mean()
        )

        return data

    def get_required_warmup_period(self) -> int:
        """
        Get the required warmup period for the strategy.
        For MA strategy, we need max(long_window, short_window) periods
        """
        return max(self.long_window, self.short_window)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on moving average crossovers.

        The strategy generates:
            1 (Buy)  : When short-term MA crosses above long-term MA
            -1 (Sell): When short-term MA crosses below long-term MA
            0 (Hold) : When no crossing occurs

        Args:
            data (pd.DataFrame): Historical price data with required columns:
                - close: Closing prices

        Returns:
            pd.Series: Trading signals aligned with the input data's index
        """
        # No need to handle warmup here since it's handled by prepare_data
        signals = pd.Series(0, index=data.index)

        # Generate signals using the clean data
        signals[data["SMA_short"] > data["SMA_long"]] = 1
        signals[data["SMA_short"] < data["SMA_long"]] = -1

        return signals

    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.

        Checks:
            1. Windows are positive integers
            2. Short window is less than long window
            3. Windows are appropriate for the intended trading frequency

        Returns:
            bool: True if parameters are valid, False otherwise
        """
        if self.short_window <= 0:  # Add validation for zero
            raise ValueError("short_window must be greater than 0")
        if self.long_window <= 0:
            raise ValueError("long_window must be greater than 0")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")
        return True

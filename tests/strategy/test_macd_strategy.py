import pytest
import numpy as np
import pandas as pd
from src.strategy.macd_strategy import MACDStrategy


class TestMACDStrategy:
    @pytest.fixture
    def strategy(self):
        return MACDStrategy({"fast_period": 12, "slow_period": 26, "signal_period": 9})

    @pytest.fixture
    def sample_data(self):
        """Create sample price data with known patterns"""
        np.random.seed(42)
        periods = 100

        # Generate trending price data
        base = np.linspace(100, 200, periods)
        noise = np.random.normal(0, 2, periods)
        prices = base + noise

        df = pd.DataFrame(
            {
                "close": prices,
                "timestamp": pd.date_range(
                    start="2024-01-01", periods=periods, freq="h"
                ),
            }
        )
        return df

    def test_calculate_indicators(self, strategy, sample_data):
        """Test that MACD indicators are calculated correctly"""
        data = strategy.calculate_indicators(sample_data)

        # Check that all indicators exist
        assert "MACD" in data.columns
        assert "Signal" in data.columns
        assert "Histogram" in data.columns

        # Verify calculations for a specific point
        test_idx = 50

        # Calculate expected values manually
        fast_ema = sample_data["close"].ewm(span=12, adjust=False).mean()
        slow_ema = sample_data["close"].ewm(span=26, adjust=False).mean()
        expected_macd = fast_ema - slow_ema
        expected_signal = expected_macd.ewm(span=9, adjust=False).mean()

        assert abs(data["MACD"].iloc[test_idx] - expected_macd.iloc[test_idx]) < 0.0001
        assert (
            abs(data["Signal"].iloc[test_idx] - expected_signal.iloc[test_idx]) < 0.0001
        )
        assert (
            abs(
                data["Histogram"].iloc[test_idx]
                - (expected_macd.iloc[test_idx] - expected_signal.iloc[test_idx])
            )
            < 0.0001
        )

    def test_generate_signals(self, strategy, sample_data):
        """Test signal generation logic"""
        # Calculate indicators first
        data = strategy.calculate_indicators(sample_data)
        signals = strategy.generate_signals(data)

        # Verify signals are valid
        assert isinstance(signals, pd.Series)
        assert signals.isin([-1, 0, 1]).all()
        assert len(signals) == len(sample_data)

        # Verify signal logic
        for i in range(len(signals)):
            if data["MACD"].iloc[i] > 0:
                assert signals.iloc[i] == 1
            elif data["MACD"].iloc[i] < 0:
                assert signals.iloc[i] == -1
            else:
                assert signals.iloc[i] == 0

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        strategy = MACDStrategy(
            {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        )
        assert strategy.validate_parameters() == True

        # Test invalid parameters
        with pytest.raises(ValueError):
            MACDStrategy({"fast_period": 26, "slow_period": 12, "signal_period": 9})

        with pytest.raises(ValueError):
            MACDStrategy({"fast_period": 0, "slow_period": 26, "signal_period": 9})

        with pytest.raises(ValueError):
            MACDStrategy({"fast_period": 12, "slow_period": 26, "signal_period": 0})

    def test_warmup_period(self, strategy):
        """Test warmup period calculation"""
        # Should be max(slow_period, fast_period + signal_period)
        expected_warmup = max(26, 12 + 9)
        assert strategy.get_required_warmup_period() == expected_warmup

    def test_default_parameters(self):
        """Test default parameter initialization"""
        strategy = MACDStrategy({})
        assert strategy.fast_period == 12
        assert strategy.slow_period == 26
        assert strategy.signal_period == 9

    def test_edge_cases(self, strategy):
        """Test edge cases with minimal data"""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        with pytest.raises(
            KeyError
        ):  # Should raise error due to missing 'close' column
            strategy.calculate_indicators(empty_df)

        # Test with single row
        single_row = pd.DataFrame({"close": [100]})
        result = strategy.calculate_indicators(single_row)
        assert not pd.isna(result["MACD"].iloc[0])
        assert not pd.isna(result["Signal"].iloc[0])
        assert not pd.isna(result["Histogram"].iloc[0])

    def test_crossover_signals(self):
        """Test specific crossover scenarios"""
        strategy = MACDStrategy(
            {"fast_period": 2, "slow_period": 4, "signal_period": 2}
        )

        # Create rising market scenario
        df = pd.DataFrame({"close": [10, 11, 12, 13, 14, 15]})  # Rising prices

        data = strategy.calculate_indicators(df)
        signals = strategy.generate_signals(data)

        # In a rising market, we should see a buy signal
        assert 1 in signals.values

        # Create falling market scenario
        df = pd.DataFrame({"close": [15, 14, 13, 12, 11, 10]})  # Falling prices

        data = strategy.calculate_indicators(df)
        signals = strategy.generate_signals(data)

        # In a falling market, we should see a sell signal
        assert -1 in signals.values

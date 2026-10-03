import numpy as np
import pandas as pd
import pytest

from src.engine.backtest_engine import BacktestEngine
from src.strategy.regime_strategy import DOWNTREND, RANGE, UPTREND, RegimeStrategy
from src.strategy.strategy_factory import StrategyFactory


def make_bars(close: np.ndarray, start: str = "2023-01-01") -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "time_key": pd.date_range(start, periods=len(close), freq="D").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        }
    )


@pytest.fixture
def strategy():
    return RegimeStrategy({})


def test_registered_in_factory():
    assert isinstance(StrategyFactory.create_strategy("regime", {}), RegimeStrategy)
    assert "regime" in StrategyFactory.get_available_strategies()


def test_uptrend_holds_position(strategy):
    data = strategy.calculate_indicators(make_bars(np.linspace(100, 200, 200)))
    tail = data.iloc[strategy.get_required_warmup_period() :]
    assert (tail["Regime"] == UPTREND).all()
    assert (strategy.generate_signals(data).loc[tail.index] == 1).all()


def test_uptrend_without_entry_only_holds():
    strategy = RegimeStrategy({"enter_on_uptrend": False})
    data = strategy.calculate_indicators(make_bars(np.linspace(100, 200, 200)))
    tail = data.index[strategy.get_required_warmup_period() :]
    assert (strategy.generate_signals(data).loc[tail] == 0).all()


def test_downtrend_liquidates(strategy):
    data = strategy.calculate_indicators(make_bars(np.linspace(200, 100, 200)))
    tail = data.iloc[strategy.get_required_warmup_period() :]
    assert (tail["Regime"] == DOWNTREND).all()
    assert (strategy.generate_signals(data).loc[tail.index] == -1).all()


def test_slow_decline_with_low_adx_liquidates(strategy):
    noise = np.random.default_rng(1).normal(0, 1.5, 200)
    data = strategy.calculate_indicators(make_bars(np.linspace(90, 50, 200) + noise))
    tail = data.iloc[strategy.get_required_warmup_period() :]
    down = tail[tail["Regime"] == DOWNTREND]

    assert (down["ADX"] <= strategy.adx_threshold).any()
    assert len(down) > 0.8 * len(tail)
    assert (strategy.generate_signals(data).loc[down.index] == -1).all()


def test_range_buys_low_and_sells_high(strategy):
    t = np.arange(300)
    noise = np.random.default_rng(0).normal(0, 1, len(t))
    close = 100 + 4 * np.sin(2 * np.pi * t / 30) + noise
    data = strategy.calculate_indicators(make_bars(close))
    tail = data.iloc[strategy.get_required_warmup_period() :]
    signals = strategy.generate_signals(data).loc[tail.index]

    assert (tail["Regime"] == RANGE).mean() > 0.9
    in_range = tail["Regime"] == RANGE
    buys = tail[in_range & (signals == 1)]
    sells = tail[in_range & (signals == -1)]
    assert not buys.empty and not sells.empty
    assert (buys["close"] < buys["BB_Middle"]).all()
    assert (sells["close"] > sells["BB_Middle"]).all()


def test_no_signals_before_indicators_are_ready(strategy):
    data = strategy.calculate_indicators(make_bars(np.linspace(100, 200, 200)))
    signals = strategy.generate_signals(data)
    first_ready = strategy.ma_period + strategy.ma_slope_period - 1
    assert (signals.iloc[:first_ready] == 0).all()
    assert signals.iloc[first_ready] == 1


@pytest.mark.parametrize(
    "params",
    [
        {"adx_period": 0},
        {"ma_period": -1},
        {"bb_std": 0},
        {"down_slope_pct": 0},
        {"adx_threshold": 0},
        {"rsi_oversold": 70, "rsi_overbought": 30},
    ],
)
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        RegimeStrategy(params)


def test_engine_run_up_range_down():
    up = np.linspace(100, 160, 120)
    t = np.arange(120)
    rng = 160 + 4 * np.sin(2 * np.pi * t / 30)
    down = np.linspace(160, 100, 120)
    bars = make_bars(np.concatenate([up, rng, down]), start="2022-09-01")

    engine = BacktestEngine(RegimeStrategy({}), initial_capital=100_000.0, lot_size=1)
    report = engine.run(
        bars,
        "US.TEST",
        start_date=pd.Timestamp("2022-12-01"),
        end_date=pd.Timestamp("2023-12-31"),
    )

    portfolio = engine.portfolio
    assert portfolio.index[0] >= pd.Timestamp("2022-12-01")
    assert portfolio["position"].iloc[-1] == 0
    assert report is not None

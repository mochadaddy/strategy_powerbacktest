from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data.data_store import DataStore
from src.engine.backtest_runner import BacktestConfig, BacktestRunner


def make_bars(start: str, end: str) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    n = len(dates)
    return pd.DataFrame(
        {
            "code": "HK.00700",
            "time_key": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "open": np.linspace(100, 110, n),
            "close": np.linspace(101, 111, n),
            "high": np.linspace(102, 112, n),
            "low": np.linspace(99, 109, n),
            "volume": np.arange(n, dtype=np.int64) * 1000,
        }
    )


@pytest.fixture
def store(tmp_path):
    return DataStore(db_path=str(tmp_path / "cache" / "market_data.db"))


def test_load_range_requires_coverage(store):
    bars = make_bars("2023-01-02", "2023-03-31")
    store.save_data("HK.00700", bars, "DAY")

    assert (
        store.load_range("HK.00700", "DAY", datetime(2023, 1, 2), datetime(2023, 3, 31))
        is None
    )

    store.mark_covered("HK.00700", "DAY", datetime(2023, 1, 1), datetime(2023, 3, 31))
    loaded = store.load_range(
        "HK.00700", "DAY", datetime(2023, 1, 1), datetime(2023, 3, 31)
    )
    pd.testing.assert_frame_equal(loaded, bars)


def test_load_range_filters_sub_range(store):
    store.save_data("HK.00700", make_bars("2023-01-02", "2023-03-31"), "DAY")
    store.mark_covered("HK.00700", "DAY", datetime(2023, 1, 1), datetime(2023, 3, 31))

    loaded = store.load_range(
        "HK.00700", "DAY", datetime(2023, 2, 1), datetime(2023, 2, 28)
    )

    assert loaded["time_key"].iloc[0] == "2023-02-01 00:00:00"
    assert loaded["time_key"].iloc[-1] == "2023-02-28 00:00:00"
    assert (
        store.load_range("HK.00700", "DAY", datetime(2023, 2, 1), datetime(2023, 4, 30))
        is None
    )
    assert (
        store.load_range("HK.00700", "60M", datetime(2023, 2, 1), datetime(2023, 2, 28))
        is None
    )


def test_mark_covered_merges_adjacent_ranges(store):
    store.mark_covered("HK.00700", "DAY", datetime(2023, 1, 1), datetime(2023, 6, 30))
    store.mark_covered("HK.00700", "DAY", datetime(2023, 7, 1), datetime(2023, 12, 31))
    store.mark_covered("HK.00700", "DAY", datetime(2024, 3, 1), datetime(2024, 3, 31))

    assert store.is_covered(
        "HK.00700", "DAY", datetime(2023, 3, 1), datetime(2023, 10, 1)
    )
    assert not store.is_covered(
        "HK.00700", "DAY", datetime(2023, 12, 1), datetime(2024, 3, 15)
    )


def test_mark_covered_excludes_today(store):
    today = pd.Timestamp.today().normalize()
    store.mark_covered("HK.00700", "DAY", today - pd.Timedelta(days=30), today)

    assert store.is_covered(
        "HK.00700", "DAY", today - pd.Timedelta(days=30), today - pd.Timedelta(days=1)
    )
    assert not store.is_covered("HK.00700", "DAY", today - pd.Timedelta(days=30), today)


def test_save_data_serializes_timestamps(store):
    bars = make_bars("2023-01-02", "2023-01-31")
    bars["time_key"] = pd.to_datetime(bars["time_key"])
    store.save_data("HK.00700", bars, "4H")
    store.mark_covered("HK.00700", "4H", datetime(2023, 1, 1), datetime(2023, 1, 31))

    loaded = store.load_range(
        "HK.00700", "4H", datetime(2023, 1, 1), datetime(2023, 1, 31)
    )

    assert loaded["time_key"].iloc[0] == "2023-01-02 00:00:00"
    assert len(loaded) == len(bars)


def test_lot_size_cache(store):
    assert store.load_lot_size("HK.00700") is None
    store.save_lot_size("HK.00700", 100)
    assert store.load_lot_size("HK.00700") == 100


class FakeFetcher:
    def __init__(self):
        self.fetch_calls = 0
        self.lot_size_calls = 0

    def fetch_data(self, symbol, start_date, end_date, timeframe, warmup_periods=0):
        self.fetch_calls += 1
        start = start_date - pd.Timedelta(days=100)
        return make_bars(start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    def fetch_lot_size(self, symbol):
        self.lot_size_calls += 1
        return 100


def make_config(start: datetime, end: datetime) -> BacktestConfig:
    return BacktestConfig(
        strategy_name="macd",
        strategy_params={},
        symbols=["HK.00700"],
        start_date=start,
        end_date=end,
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0,
        timeframe="DAY",
    )


def test_runner_fetches_from_opend_only_on_cache_miss(tmp_path):
    runner = BacktestRunner(db_path=str(tmp_path / "market_data.db"))
    fetcher = FakeFetcher()
    runner._data_fetcher = fetcher

    first = runner.load_symbol_data(
        "HK.00700", make_config(datetime(2023, 1, 1), datetime(2023, 12, 31)), 26
    )
    assert fetcher.fetch_calls == 1

    second_runner = BacktestRunner(db_path=str(tmp_path / "market_data.db"))
    second = second_runner.load_symbol_data(
        "HK.00700", make_config(datetime(2023, 1, 1), datetime(2023, 12, 31)), 26
    )
    assert second_runner._data_fetcher is None
    pd.testing.assert_frame_equal(first, second)

    sub_range = second_runner.load_symbol_data(
        "HK.00700", make_config(datetime(2023, 6, 1), datetime(2023, 9, 30)), 26
    )
    assert second_runner._data_fetcher is None
    assert sub_range["time_key"].iloc[-1] == "2023-09-29 00:00:00"

    runner.load_symbol_data(
        "HK.00700", make_config(datetime(2023, 6, 1), datetime(2024, 3, 31)), 26
    )
    assert fetcher.fetch_calls == 2


def test_runner_caches_lot_size(tmp_path):
    runner = BacktestRunner(db_path=str(tmp_path / "market_data.db"))
    fetcher = FakeFetcher()
    runner._data_fetcher = fetcher

    assert runner.load_lot_size("HK.00700") == 100
    assert runner.load_lot_size("HK.00700") == 100
    assert fetcher.lot_size_calls == 1

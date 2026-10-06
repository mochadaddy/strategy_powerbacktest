from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd
import os

from ..data.data_fetcher import FutuDataFetcher
from ..strategy.strategy_factory import StrategyFactory
from .backtest_engine import BacktestEngine
from ..utils.logger import setup_logger
from ..data.data_store import DataStore


@dataclass
class BacktestConfig:
    """Configuration for backtest run"""

    strategy_name: str
    strategy_params: Dict
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission: float
    slippage: float
    timeframe: str


class BacktestRunner:
    """
    Handles the execution of backtests with different strategies and parameters

    Attributes:
        data_fetcher (FutuDataFetcher): Data fetcher instance
        logger: Logger instance
        BacktestRunner.__init__() FutuDataFetcher初始化针对特定 Futu OpenD 主机和端口的底层数据获取器实例（ ），以及专用日志记录器。
    """

    def __init__(self, host: str = "localhost", port: int = 11111,db_path: Optional[str] = None):
        """
        Initialize BacktestRunner

        Args:
            host (str): Futu OpenD host
            port (int): Futu OpenD port
        """
        self.host = host
        self.port = port
        self._data_fetcher: Optional[FutuDataFetcher] = None
        self.data_store = DataStore(db_path=db_path)
        self.logger = setup_logger(__name__)

    @property
    def data_fetcher(self) -> FutuDataFetcher:
        """Connect to Futu OpenD only when data is missing from the local cache"""
        if self._data_fetcher is None:
            self._data_fetcher = FutuDataFetcher(host=self.host, port=self.port)
        return self._data_fetcher

    def load_symbol_data(
            self, symbol: str, config: BacktestConfig, warmup_periods: int
    ) -> pd.DataFrame:
        """Load data from the SQLite cache, falling back to Futu OpenD on a miss"""
        fetch_start = FutuDataFetcher.get_fetch_start(config.start_date)
        cached = self.data_store.load_range(
            symbol, config.timeframe, fetch_start, config.end_date
        )
        if cached is not None and not cached.empty:
            self.logger.info(
                f"[{symbol}] Loaded {len(cached)} {config.timeframe} bars from cache"
            )
            return cached

        self.logger.info(f"[{symbol}] Cache miss, fetching from Futu OpenD")
        df = self.data_fetcher.fetch_data(
            symbol=symbol,
            start_date=config.start_date,
            end_date=config.end_date,
            timeframe=config.timeframe,
            warmup_periods=warmup_periods,
        )
        self.data_store.save_data(symbol, df, config.timeframe)
        self.data_store.mark_covered(
            symbol, config.timeframe, fetch_start, config.end_date
        )
        return df

    def load_lot_size(self, symbol: str) -> int:
        """Load lot size from the SQLite cache, falling back to Futu OpenD on a miss"""
        lot_size = self.data_store.load_lot_size(symbol)
        if lot_size is not None:
            return lot_size

        lot_size = self.data_fetcher.fetch_lot_size(symbol)
        if lot_size is None:
            return 1
        self.data_store.save_lot_size(symbol, lot_size)
        return lot_size
    def run(self, config: BacktestConfig) -> str:
        """
        Run backtest with multiple symbols

        Args:
            config (BacktestConfig): Backtest configuration

        Returns:
            str: Path to generated report

        Raises:
            ValueError: If strategy creation fails

        """
        # Create strategy

        # 策略实例化：StrategyFactory.create_strategy()使用config.strategy_name和进行调用config.strategy_params。

        strategy = StrategyFactory.create_strategy(
            config.strategy_name, config.strategy_params
        )
        # 预热期计算：查询strategy.get_required_warmup_period()以确定之前所需的历史数据start_date。
        warmup_periods = strategy.get_required_warmup_period()

        # Fetch data for all symbols in parallel
        data_dict = {
            symbol: self.load_symbol_data(symbol, config, warmup_periods)
            for symbol in config.symbols
        }
        lot_size = self.load_lot_size(config.symbols[-1])

        # Initialize and run backtest engine
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=config.initial_capital,
            commission=config.commission,
            lot_size=lot_size,
        )

        results = engine.run_multi_symbol(data_dict, config.start_date, config.end_date)

        # Generate report
        output_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(output_dir, exist_ok=True)
        return results.generate_report(output_dir)

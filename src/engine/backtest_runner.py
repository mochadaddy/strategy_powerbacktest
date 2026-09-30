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
        self.data_fetcher = FutuDataFetcher(host=host, port=port)
        self.data_store = DataStore(db_path=db_path)  # 新增
        self.logger = setup_logger(__name__)

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
        data_dict = {}
        for symbol in config.symbols:
            df = self.data_store.load_data(symbol, config.timeframe)
            if df is None or df.empty:
                # 2. 未命中 → 向 Futu 拉取
                df = self.data_fetcher.fetch_data(
                    symbol=symbol,
                    start_date=config.start_date,
                    end_date=config.end_date,
                    timeframe=config.timeframe,
                    warmup_periods=warmup_periods,
                )
                self.data_store.save_data(symbol, df, config.timeframe)
                data_dict[symbol] = df
            data_dict[symbol] = self.data_fetcher.fetch_data(
                symbol=symbol,
                start_date=config.start_date,
                end_date=config.end_date,
                timeframe=config.timeframe,
                warmup_periods=warmup_periods,
            )

        # Fetch lot size and fundamental data
        lot_size = self.data_fetcher.fetch_lot_size(symbol)

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
        results.generate_report(output_dir)

        return os.path.join(output_dir, "strategy_backtest_report.html")

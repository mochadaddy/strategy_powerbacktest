from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime
from typing import Dict, Any
from .config import Config


def parse_date(date_str: str) -> datetime:
    """Convert date string to datetime object"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format")

# 获取命令参数
def create_cli_parser() -> ArgumentParser:
    """Create command line argument parser with all available options"""
    parser = ArgumentParser(
        description="Strategy Power Backtester - A professional-grade algorithmic trading backtesting framework",
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Strategy selection
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy to use for backtesting (e.g., macd, ma_cross)",
    )

    # Symbol selection
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of symbols to backtest (e.g., AAPL MSFT GOOG)",
    )

    # Date range
    parser.add_argument(
        "--start-date", type=parse_date, help="Start date for backtesting (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date", type=parse_date, help="End date for backtesting (YYYY-MM-DD)"
    )

    # Capital and commission
    parser.add_argument(
        "--initial-capital", type=float, help="Initial capital for backtesting"
    )

    parser.add_argument(
        "--commission", type=float, help="Commission rate (e.g., 0.001 for 0.1%%)"
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="DAY",
        choices=[
            "1M",
            "3M",
            "5M",
            "15M",
            "30M",
            "60M",
            "2H",
            "4H",
            "DAY",
            "WEEK",
            "MON",
        ],
        help="K-line timeframe for backtesting",
    )

    return parser


def get_backtest_config(args: Any, config: Config) -> Dict[str, Any]:
    """
    Merge command line arguments with config file settings to create backtest configuration
    """
    # Start with default values from config file
    backtest_params = (
        config.backtest_config.copy() if hasattr(config, "backtest_config") else {}
    )

    # Override with command line arguments if provided
    if args.symbols:
        backtest_params["symbols"] = args.symbols

    if args.strategy:
        backtest_params["strategy_name"] = args.strategy

    if args.start_date:
        if isinstance(args.start_date, datetime):
            backtest_params["start_date"] = args.start_date
        else:
            backtest_params["start_date"] = datetime.strptime(
                args.start_date, "%Y-%m-%d"
            )

    if args.end_date:
        if isinstance(args.end_date, datetime):
            backtest_params["end_date"] = args.end_date
        else:
            backtest_params["start_date"] = datetime.strptime(
                args.start_date, "%Y-%m-%d"
            )

    if args.initial_capital:
        backtest_params["initial_capital"] = args.initial_capital

    if args.commission:
        backtest_params["commission"] = args.commission

    if args.timeframe:
        backtest_params["timeframe"] = args.timeframe

    # Add strategy parameters if provided in config
    strategy_params = config.get_strategy_params(backtest_params["strategy_name"])
    backtest_params["strategy_params"] = strategy_params if strategy_params else {}

    return backtest_params

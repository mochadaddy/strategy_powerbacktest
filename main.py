from datetime import datetime
from src.engine.backtest_runner import BacktestRunner, BacktestConfig
from src.strategy.strategy_factory import StrategyFactory
from src.utils.config import Config
from src.utils.cli_parser import create_cli_parser, get_backtest_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    # Parse command line arguments
    parser = create_cli_parser()
    args = parser.parse_args()

    # Load configuration
    config = Config("config.yaml")

    # Display available strategies
    available_strategies = StrategyFactory.get_available_strategies()
    logger.info("Available strategies:")
    for name, info in available_strategies.items():
        logger.info(f"- {info['name']}: {info['description']}")

    # Get merged configuration
    backtest_params = get_backtest_config(args, config)

    # Initialize backtest runner
    runner = BacktestRunner(
        host=config.futu_config["host"], port=config.futu_config["port"],db_path=config.data_config.get("storage", {}).
        get("db_path")
    )

    # Create and run backtest
    backtest_config = BacktestConfig(**backtest_params)
    report_path = runner.run(backtest_config)
    logger.info(f"Backtest report generated at: {report_path}")


if __name__ == "__main__":
    main()

from typing import Dict, Type
from .base_strategy import BaseStrategy
from .macd_strategy import MACDStrategy
from .moving_average_strategy import MovingAverageCrossStrategy
from .btse_strategy import BTSEStrategy
from .leg_strategy import LEGStrategy

# 维护策略键与其对应实现类之间的静态映射，并为 CLI 和配置验证提供参数元数据。
class StrategyFactory:
    """
    Factory class for creating trading strategies

    Attributes:
        _strategies (Dict[str, Type[BaseStrategy]]): Registry of available strategies
    """

    _strategies = {
        "macd": MACDStrategy,
        "ma_cross": MovingAverageCrossStrategy,
        "btse": BTSEStrategy,
        "leg": LEGStrategy
    }

    @classmethod
    def get_available_strategies(cls) -> Dict[str, Dict]:
        """
        Get list of available strategies with their descriptions and parameters

        Returns:
            Dict[str, Dict]: Dictionary of strategy information
        """
        return {
            "macd": {
                "name": "MACD Strategy",
                "description": "Moving Average Convergence Divergence strategy",
                "parameters": {
                    "fast_period": {"type": "int", "default": 12, "min": 2, "max": 50},
                    "slow_period": {"type": "int", "default": 26, "min": 5, "max": 100},
                    "signal_period": {"type": "int", "default": 9, "min": 2, "max": 50},
                },
            },
            "ma_cross": {
                "name": "Moving Average Crossover",
                "description": "Simple moving average crossover strategy",
                "parameters": {
                    "short_window": {
                        "type": "int",
                        "default": 20,
                        "min": 1,
                        "max": 100,
                    },
                    "long_window": {
                        "type": "int",
                        "default": 50,
                        "min": 10,
                        "max": 200,
                    },
                },
            },
            "btse": {
                "name": "BTSE Strategy",
                "description": "Breakthrough Strength Evaluation strategy combining multiple technical indicators",
                "parameters": {
                    "lookback_period": {
                        "type": "int",
                        "default": 25,
                        "min": 10,
                        "max": 100,
                    },
                    "breakthrough_threshold": {
                        "type": "int",
                        "default": 3,
                        "min": 1,
                        "max": 10,
                    },
                    "ma_period": {"type": "int", "default": 2, "min": 1, "max": 10},
                },
            },
            "leg": {
                "name": "LEG Strategy",
                "description": "Level Entry Guide strategy combining multiple technical indicators",
                "parameters": {
                    "ma_period": {"type": "int", "default": 17, "min": 1, "max": 100},
                    "vol_period": {"type": "int", "default": 88, "min": 1, "max": 100},
                    "rsi_period": {"type": "int", "default": 9, "min": 1, "max": 100},
                    "pattern_lookback": {
                        "type": "int",
                        "default": 120,
                        "min": 1,
                        "max": 1000,
                    },
                },
            },
        }

    @classmethod
    def create_strategy(cls, strategy_name: str, parameters: Dict) -> BaseStrategy:
        """
        Create a strategy instance with given parameters

        Args:
            strategy_name (str): Name of the strategy to create
            parameters (Dict): Strategy parameters

        Returns:
            BaseStrategy: Initialized strategy instance

        Raises:
            ValueError: If strategy_name is not found in registry
        """
        if strategy_name not in cls._strategies:
            raise ValueError(
                f"Strategy '{strategy_name}' not found. Available strategies: {list(cls._strategies.keys())}"
            )

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(parameters)

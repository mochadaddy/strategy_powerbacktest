from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
from logging import Logger
from src.strategy.base_strategy import BaseStrategy
from src.utils.logger import setup_logger
from .metrics.return_metrics import ReturnMetrics
from .metrics.risk_metrics import RiskMetrics
from .metrics.trade_metrics import TradeMetrics
from .strategy_backtest_report import StrategyBacktestReport
from .symbol_results import SymbolResults, BacktestReport
from .benchmark_portfolio import BenchmarkPortfolio


@dataclass
class TradeExecution:
    """Data class for trade execution details"""

    timestamp: datetime
    type: str
    price: float
    quantity: int
    commission: float
    pnl: Optional[float] = None
    cost: Optional[float] = None
    proceeds: Optional[float] = None


class BacktestEngine:
    """
    Backtesting engine for simulating trading strategies.

    This class provides a framework for testing trading strategies with historical data,
    handling position sizing, trade execution, and performance tracking.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        lot_size: int = 1,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize backtesting engine.

        Args:
            strategy: Trading strategy implementation
            initial_capital: Starting capital for backtest
            commission: Trading commission rate
            logger: Optional custom logger instance
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.portfolio = pd.DataFrame()
        self.trades: List[TradeExecution] = []
        self.logger = logger or setup_logger(__name__)
        self.lot_size = lot_size
        self.symbol: Optional[str] = None

    def run(
        self, data: pd.DataFrame, symbol: str, start_date: datetime, end_date: datetime
    ) -> BacktestReport:
        """Execute backtest for the given data and symbol."""
        self.symbol = symbol
        self._log_backtest_start(data, symbol)

        # Calculate indicators with warmup period
        full_data, trading_data = self.strategy.prepare_data(
            data, start_date=start_date, end_date=end_date
        )

        # Generate signals only on the trading data (post-warmup)
        signals = self.strategy.generate_signals(trading_data)

        # Calculate portfolio metrics excluding warmup period
        self.portfolio = self._calculate_portfolio(trading_data, signals)
        self.trades = self._generate_trades_list(trading_data, signals)

        # Calculate comprehensive backtest metrics
        metrics = self._calculate_metrics()

        # Calculate benchmark data
        benchmark_portfolio = BenchmarkPortfolio(self.initial_capital, self.lot_size)
        metrics["benchmark_data"] = benchmark_portfolio.calculate_buy_and_hold(data)

        # Log final results
        self._log_backtest_summary(metrics)

        # Generate backtest report
        return BacktestReport.from_backtest_results(
            portfolio=self.portfolio,
            trades=self.trades,
            initial_capital=self.initial_capital,
            metrics=metrics,
        )

    def _log_backtest_start(self, data: pd.DataFrame, symbol: str) -> None:
        """Log backtest initialization details"""
        start_date = pd.to_datetime(data["time_key"].iloc[0]).strftime("%Y-%m-%d")
        end_date = pd.to_datetime(data["time_key"].iloc[-1]).strftime("%Y-%m-%d")

        self.logger.info("=" * 50)
        self.logger.info("Backtest Configuration:")
        self.logger.info(f"Symbol: {symbol} (Lot Size: {self.lot_size})")
        self.logger.info(f"Period: {start_date} to {end_date}")
        self.logger.info(f"Strategy: {self.strategy.__class__.__name__}")
        self.logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"Commission Rate: {self.commission*100:.2f}%")
        self.logger.info("=" * 50)

    def _log_trade_execution(self, trade: Dict[str, Any]) -> None:
        """Log trade execution details"""
        trade_type = trade["type"].upper()
        quantity = trade["quantity"]
        lots = quantity // self.lot_size
        price = trade["price"]

        if trade_type == "BUY":
            cost = trade["cost"]
            commission = trade["commission"]
            self.logger.info(
                f"TRADE: {trade['timestamp']} | {trade_type} | "
                f"{quantity:,d} units ({lots:,d} lots) @ ${price:.3f} | "
                f"Cost: ${cost:,.2f} | Commission: ${commission:.2f}"
            )
        else:  # SELL
            proceeds = trade["proceeds"]
            pnl = trade["pnl"]
            self.logger.info(
                f"TRADE: {trade['timestamp']} | {trade_type} | "
                f"{quantity:,d} units ({lots:,d} lots) @ ${price:.3f} | "
                f"Proceeds: ${proceeds:,.2f} | PnL: ${pnl:,.2f}"
            )

    def _log_backtest_summary(self, metrics: Dict[str, Any]) -> None:
        """Log final backtest results"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info(f"Backtest Summary {self.symbol}:")
        self.logger.info(f"Total Return: {metrics['total_return']*100:.2f}%")
        self.logger.info(f"Annual Return: {metrics['annual_return']*100:.2f}%")
        self.logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        self.logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        self.logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        self.logger.info(f"Total Trades: {metrics['total_trades']}")
        self.logger.info(f"Total PnL: ${metrics['total_pnl']:,.2f}")
        self.logger.info("=" * 50)

    def _calculate_portfolio(
        self, data: pd.DataFrame, signals: pd.Series
    ) -> pd.DataFrame:
        """Calculate portfolio value over time with lot size handling"""
        # Initialize portfolio DataFrame properly
        portfolio = pd.DataFrame(
            {
                "time_key": data["time_key"],
                "position": 0,
                "close": pd.to_numeric(data["close"].values),
                "cash": self.initial_capital,
            }
        ).set_index("time_key")
        portfolio.index = pd.to_datetime(portfolio.index)

        current_position = 0
        for i in range(1, len(signals)):
            price = portfolio["close"].iloc[i]

            if signals[i] == 1 and current_position == 0:  # Buy signal
                max_shares = int(
                    portfolio["cash"].iloc[i - 1] / (price * (1 + self.commission))
                )
                lots_to_buy = (max_shares // self.lot_size) * self.lot_size

                if lots_to_buy >= self.lot_size:
                    cost = lots_to_buy * price * (1 + self.commission)
                    current_position = lots_to_buy
                    portfolio.loc[portfolio.index[i:], "position"] = lots_to_buy
                    portfolio.loc[portfolio.index[i], "cash"] = (
                        portfolio["cash"].iloc[i - 1] - cost
                    )

            elif signals[i] == -1 and current_position > 0:  # Sell signal
                proceeds = current_position * price * (1 - self.commission)
                current_position = 0
                portfolio.loc[portfolio.index[i:], "position"] = 0
                portfolio.loc[portfolio.index[i], "cash"] = (
                    portfolio["cash"].iloc[i - 1] + proceeds
                )
            else:
                portfolio.loc[portfolio.index[i], "cash"] = portfolio["cash"].iloc[
                    i - 1
                ]

        # Use proper forward fill
        portfolio["cash"] = portfolio["cash"].ffill()

        # Calculate portfolio values
        portfolio["holdings"] = portfolio["position"] * portfolio["close"]
        portfolio["total"] = portfolio["cash"] + portfolio["holdings"]

        # Calculate returns properly
        portfolio.loc[:, "returns"] = portfolio["total"].pct_change()
        portfolio.loc[portfolio.index[0], "returns"] = 0

        return portfolio

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics using metric classes"""
        final_value = self.portfolio["total"].iloc[-1]
        trades_with_pnl = [t for t in self.trades if t.get("pnl") is not None]

        # Return metrics - calculate total_return first
        total_return = ReturnMetrics.calculate_total_return(
            final_value, self.initial_capital
        )

        return_metrics = {
            "total_return": total_return,  # Now we can use total_return
            "annual_return": ReturnMetrics.calculate_annual_return(
                self.portfolio, total_return
            ),
            "sharpe_ratio": ReturnMetrics.calculate_sharpe_ratio(self.portfolio),
            "monthly_returns": ReturnMetrics.calculate_monthly_returns(self.portfolio),
        }

        # Risk metrics
        risk_metrics = {
            "volatility": RiskMetrics.calculate_volatility(self.portfolio),
            "max_drawdown": RiskMetrics.calculate_max_drawdown(self.portfolio),
            "max_drawdown_duration": RiskMetrics.calculate_drawdown_duration(
                self.portfolio
            ),
            "value_at_risk": RiskMetrics.calculate_value_at_risk(self.portfolio),
            "beta": RiskMetrics.calculate_beta(self.portfolio),
            "sortino_ratio": RiskMetrics.calculate_sortino_ratio(self.portfolio),
        }

        # Trade metrics
        trade_metrics = {
            **TradeMetrics.calculate_trade_counts(trades_with_pnl),
            "win_rate": TradeMetrics.calculate_win_rate(trades_with_pnl),
            "profit_factor": TradeMetrics.calculate_profit_factor(trades_with_pnl),
            **TradeMetrics.calculate_trade_stats(trades_with_pnl),
            "max_consecutive_wins": TradeMetrics.calculate_consecutive_stats(
                trades_with_pnl, "wins"
            ),
            "max_consecutive_losses": TradeMetrics.calculate_consecutive_stats(
                trades_with_pnl, "losses"
            ),
            "avg_position_duration": TradeMetrics.calculate_position_duration(
                trades_with_pnl
            ),
            # Use ReturnMetrics for PnL calculations
            "realized_pnl": ReturnMetrics.calculate_realized_pnl(trades_with_pnl),
            "floating_pnl": ReturnMetrics.calculate_floating_pnl(
                self.portfolio, trades_with_pnl
            ),
        }

        # Add total_pnl calculation
        trade_metrics["total_pnl"] = (
            trade_metrics["realized_pnl"] + trade_metrics["floating_pnl"]
        )

        # Position metrics
        position_metrics = {
            "avg_position_size": self.portfolio["position"].mean(),
            "max_position_size": self.portfolio["position"].max(),
            "current_position": self.portfolio["position"].iloc[-1],
        }

        return {
            # Strategy Info
            "strategy_name": self.strategy.__class__.__name__,
            "symbol": self.symbol,
            "lot_size": self.lot_size,
            "commission_rate": self.commission,
            # Combine all metrics
            **return_metrics,
            **risk_metrics,
            **trade_metrics,
            **position_metrics,
        }

    def _generate_trades_list(
        self, data: pd.DataFrame, signals: pd.Series
    ) -> List[Dict]:
        """Generate list of trades from signals with lot size handling"""
        trades = []
        current_position = 0
        last_buy_price = 0
        realized_pnl = 0

        for i in range(1, len(signals)):
            if signals[i] == 1 and current_position == 0:  # Buy signal
                price = data["close"].iloc[i]
                timestamp = data["time_key"].iloc[i]

                # Calculate lots to buy based on available cash
                max_shares = int(
                    self.portfolio["cash"].iloc[i - 1] / (price * (1 + self.commission))
                )
                lots_to_buy = (max_shares // self.lot_size) * self.lot_size

                if lots_to_buy >= self.lot_size:
                    cost = lots_to_buy * price * (1 + self.commission)
                    trade = {
                        "timestamp": timestamp,
                        "type": "buy",
                        "price": price,
                        "quantity": lots_to_buy,
                        "cost": cost,
                        "commission": cost - (lots_to_buy * price),
                    }
                    trades.append(trade)
                    current_position = lots_to_buy
                    last_buy_price = price

                    self.logger.info(
                        f"[{self.symbol}] Trade executed at {timestamp}: BUY "
                        f"{lots_to_buy} units ({lots_to_buy//self.lot_size} lots) at ${price:.2f} "
                        f"(Cost: ${cost:,.2f}, Commission: ${cost - (lots_to_buy * price):.2f})"
                    )

            elif signals[i] == -1 and current_position > 0:  # Sell signal
                price = data["close"].iloc[i]
                timestamp = data["time_key"].iloc[i]
                proceeds = current_position * price * (1 - self.commission)
                trade_pnl = proceeds - (
                    current_position * last_buy_price * (1 + self.commission)
                )
                realized_pnl += trade_pnl

                trade = {
                    "timestamp": timestamp,
                    "type": "sell",
                    "price": price,
                    "quantity": current_position,
                    "proceeds": proceeds,
                    "commission": (current_position * price) - proceeds,
                    "pnl": trade_pnl,
                }
                trades.append(trade)

                self.logger.info(
                    f"[{self.symbol}] Trade executed at {timestamp}: SELL "
                    f"{current_position} units ({current_position//self.lot_size} lots) at ${price:.2f} "
                    f"(Proceeds: ${proceeds:,.2f}, Commission: ${(current_position * price) - proceeds:.2f})"
                )
                self.logger.info(
                    f"Trade PnL: ${trade_pnl:,.2f} (Running PnL: ${realized_pnl:,.2f})"
                )

                current_position = 0

        # Calculate floating PnL at the end of backtest period
        if current_position > 0:
            final_price = data["close"].iloc[-1]
            floating_pnl = (final_price - last_buy_price) * current_position
            self.logger.info(
                f"\nEnd of Backtest Summary:"
                f"\nRealized PnL: ${realized_pnl:,.2f}"
                f"\nFloating PnL: ${floating_pnl:,.2f} (from {current_position} units, {current_position//self.lot_size} lots)"
                f"\nTotal PnL: ${(realized_pnl + floating_pnl):,.2f}"
            )
        else:
            self.logger.info(
                f"\nEnd of Backtest Summary:"
                f"\nRealized PnL: ${realized_pnl:,.2f}"
                f"\nNo open positions"
            )

        return trades

    def run_multi_symbol(
        self,
        data_dict: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> StrategyBacktestReport:
        """
        Execute backtest for multiple symbols.

        Args:
            data_dict: Dictionary mapping symbols to their historical data

        Returns:
            MultiSymbolBacktestReport: Consolidated report for all symbols
        """
        symbol_results = {}

        for symbol, data in data_dict.items():
            # Run individual backtest for each symbol
            report = self.run(data, symbol, start_date, end_date)
            symbol_results[symbol] = SymbolResults.from_backtest_report(report)

        # Create multi-symbol report
        return StrategyBacktestReport(
            strategy_name=self.strategy.__class__.__name__,
            start_date=pd.to_datetime(
                #start_date
                min(data.index.min() for data in data_dict.values())
            ),
            end_date=pd.to_datetime(
                #end_date
                max(data.index.max() for data in data_dict.values())
            ),
            initial_capital=self.initial_capital,
            commission_rate=self.commission,
            symbol_results=symbol_results,
        )

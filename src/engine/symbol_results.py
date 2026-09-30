from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime


@dataclass
class BacktestReport:
    """
    Comprehensive backtest report with analytics and visualization capabilities.

    This class handles the generation of performance metrics, charts, and HTML reports
    for trading strategy backtests.

    Attributes:
        Basic Info:
            strategy_name (str): Name of the trading strategy
            symbol (str): Trading instrument symbol
            start_date (datetime): Backtest start date
            end_date (datetime): Backtest end date
            lot_size (int): Trading lot size
            commission_rate (float): Trading commission rate

        Performance Metrics:
            initial_capital (float): Starting capital
            final_portfolio_value (float): Ending portfolio value
            total_return (float): Total return percentage
            annual_return (float): Annualized return percentage

        Risk Metrics:
            sharpe_ratio (float): Risk-adjusted return metric
            sortino_ratio (float): Downside risk-adjusted return metric
            max_drawdown (float): Maximum peak to trough decline
            max_drawdown_duration (int): Longest drawdown period in days
            volatility (float): Standard deviation of returns
            value_at_risk (float): 95% Value at Risk
            beta (float): Market correlation coefficient

        Trading Statistics:
            total_trades (int): Total number of trades
            winning_trades (int): Number of profitable trades
            losing_trades (int): Number of unprofitable trades
            win_rate (float): Percentage of winning trades
            profit_factor (float): Ratio of gross profits to gross losses

        Time Series Data:
            portfolio (pd.DataFrame): Portfolio value and returns history
            trades (List[Dict[str, Any]]): Detailed trade history
            monthly_returns (pd.DataFrame): Monthly return statistics
        Benchmark Data:
            benchmark_data (pd.DataFrame): Benchmark portfolio value history
    """

    # Basic Info
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    lot_size: int
    commission_rate: float

    # Capital and Returns
    initial_capital: float
    final_portfolio_value: float
    total_return: float
    annual_return: float

    # Risk Metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    value_at_risk: float
    beta: float

    # Trading Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Position Info
    avg_position_size: float
    max_position_size: float
    avg_position_duration: float

    # Time Series Data
    portfolio: pd.DataFrame
    trades: List[Dict[str, Any]]
    monthly_returns: pd.DataFrame
    benchmark_data: pd.DataFrame

    @classmethod
    def from_backtest_results(
        cls,
        portfolio: pd.DataFrame,
        trades: List[Dict[str, Any]],
        initial_capital: float,
        metrics: Dict[str, Any],
    ) -> "BacktestReport":
        """Create a backtest report from raw backtest results"""
        return cls(
            # Basic Info
            strategy_name=metrics["strategy_name"],
            symbol=metrics["symbol"],
            start_date=portfolio.index[0],
            end_date=portfolio.index[-1],
            lot_size=metrics["lot_size"],
            commission_rate=metrics["commission_rate"],
            # Capital and Returns
            initial_capital=initial_capital,
            final_portfolio_value=portfolio["total"].iloc[-1],
            total_return=metrics["total_return"],
            annual_return=metrics["annual_return"],
            # Risk Metrics
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            max_drawdown=metrics["max_drawdown"],
            max_drawdown_duration=metrics["max_drawdown_duration"],
            volatility=metrics["volatility"],
            value_at_risk=metrics["value_at_risk"],
            beta=metrics["beta"],
            # Trading Statistics
            total_trades=metrics["total_trades"],
            winning_trades=metrics["winning_trades"],
            losing_trades=metrics["losing_trades"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            avg_trade_return=metrics["avg_trade_return"],
            avg_win=metrics["avg_win"],
            avg_loss=metrics["avg_loss"],
            largest_win=metrics["largest_win"],
            largest_loss=metrics["largest_loss"],
            max_consecutive_wins=metrics["max_consecutive_wins"],
            max_consecutive_losses=metrics["max_consecutive_losses"],
            # Position Info
            avg_position_size=metrics["avg_position_size"],
            max_position_size=metrics["max_position_size"],
            avg_position_duration=metrics["avg_position_duration"],
            # Time Series Data
            portfolio=portfolio,
            trades=trades,
            monthly_returns=metrics["monthly_returns"],
            #fundamental_data=None,
            benchmark_data=metrics["benchmark_data"],
        )

    def _calculate_drawdowns(self) -> pd.Series:
        """Calculate drawdown series for plotting"""
        cumulative_returns = (1 + self.portfolio["returns"]).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return drawdowns

    def _get_metrics_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Organize metrics into groups for HTML report"""
        metrics = {
            "Strategy Info": [
                {"title": "Strategy", "value": self.strategy_name},
                {"title": "Symbol", "value": self.symbol},
                {
                    "title": "Period",
                    "value": f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
                },
                {"title": "Initial Capital", "value": f"${self.initial_capital:,.2f}"},
                {
                    "title": "Commission Rate",
                    "value": f"{self.commission_rate*100:.2f}%",
                },
            ],
            "Performance Metrics": [
                {
                    "title": "Final Portfolio Value",
                    "value": f"${self.final_portfolio_value:,.2f}",
                },
                {
                    "title": "Total Return",
                    "value": f"{self.total_return*100:.2f}%",
                    "color": "positive" if self.total_return > 0 else "negative",
                },
                {
                    "title": "Annual Return",
                    "value": f"{self.annual_return*100:.2f}%",
                    "color": "positive" if self.annual_return > 0 else "negative",
                },
                {"title": "Sharpe Ratio", "value": f"{self.sharpe_ratio:.2f}"},
                {"title": "Sortino Ratio", "value": f"{self.sortino_ratio:.2f}"},
            ],
            "Risk Metrics": [
                {
                    "title": "Max Drawdown",
                    "value": f"{self.max_drawdown*100:.2f}%",
                    "color": "negative",
                },
                {
                    "title": "Max Drawdown Duration",
                    "value": f"{self.max_drawdown_duration} days",
                },
                {"title": "Volatility", "value": f"{self.volatility*100:.2f}%"},
                {
                    "title": "Value at Risk (95%)",
                    "value": f"{self.value_at_risk*100:.2f}%",
                },
                {"title": "Beta", "value": f"{self.beta:.2f}"},
            ],
            "Trading Statistics": [
                {"title": "Total Trades", "value": str(self.total_trades)},
                {"title": "Win Rate", "value": f"{self.win_rate*100:.2f}%"},
                {"title": "Profit Factor", "value": f"{self.profit_factor:.2f}"},
                {
                    "title": "Average Trade Return",
                    "value": f"${self.avg_trade_return:,.2f}",
                    "color": "positive" if self.avg_trade_return > 0 else "negative",
                },
                {
                    "title": "Average Win",
                    "value": f"${self.avg_win:,.2f}",
                    "color": "positive",
                },
                {
                    "title": "Average Loss",
                    "value": f"${self.avg_loss:,.2f}",
                    "color": "negative",
                },
                {
                    "title": "Largest Win",
                    "value": f"${self.largest_win:,.2f}",
                    "color": "positive",
                },
                {
                    "title": "Largest Loss",
                    "value": f"${self.largest_loss:,.2f}",
                    "color": "negative",
                },
                {
                    "title": "Max Consecutive Wins",
                    "value": str(self.max_consecutive_wins),
                },
                {
                    "title": "Max Consecutive Losses",
                    "value": str(self.max_consecutive_losses),
                },
            ],
            "Position Info": [
                {
                    "title": "Average Position Size",
                    "value": f"{self.avg_position_size:,.0f} units",
                },
                {
                    "title": "Max Position Size",
                    "value": f"{self.max_position_size:,.0f} units",
                },
                {
                    "title": "Average Position Duration",
                    "value": f"{self.avg_position_duration:.1f} days",
                },
            ],
        }
        return metrics

    def _prepare_trade_annotations(self) -> List[Dict]:
        """Prepare trade annotations for visualization"""
        annotations = []
        for trade in self.trades:
            trade_type = trade["type"].upper()
            price = trade["price"]
            quantity = trade["quantity"]
            timestamp = trade["timestamp"]

            # Convert timestamp to milliseconds, handling both string and datetime
            if isinstance(timestamp, str):
                timestamp_ms = int(pd.Timestamp(timestamp).timestamp() * 1000)
            else:
                timestamp_ms = int(timestamp.timestamp() * 1000)

            annotation = {
                "x": timestamp_ms,
                "y": price,
                "type": trade_type,
                "quantity": quantity,
                "text": f"{trade_type}: {quantity} @ ${price:.2f}",
            }

            if trade_type == "SELL":
                pnl = trade.get("pnl", 0)
                annotation["text"] += f" (PnL: ${pnl:.2f})"
                annotation["color"] = "green" if pnl > 0 else "red"
            else:
                annotation["color"] = "blue"

            annotations.append(annotation)

        return annotations


@dataclass
class SymbolResults:
    """Results for individual symbol in multi-symbol backtest"""

    symbol: str
    trades: List[Dict[str, Any]]
    metrics: Dict[str, List[Dict[str, Any]]]
    equity_curve: pd.Series
    drawdown: pd.Series
    monthly_returns: pd.DataFrame
    trade_annotations: List[Dict[str, Any]]
    portfolio: pd.DataFrame
    benchmark_data: pd.DataFrame

    @classmethod
    def from_backtest_report(cls, report: BacktestReport) -> "SymbolResults":
        """Create SymbolResults from a BacktestReport instance"""
        return cls(
            symbol=report.symbol,
            trades=report.trades,
            metrics=report._get_metrics_dict(),
            equity_curve=report.portfolio["total"],
            drawdown=report._calculate_drawdowns(),
            monthly_returns=report.monthly_returns,
            trade_annotations=report._prepare_trade_annotations(),
            portfolio=report.portfolio,
            benchmark_data=report.benchmark_data,
        )

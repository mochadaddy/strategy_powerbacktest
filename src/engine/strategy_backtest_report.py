from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import os
from jinja2 import Template
from .symbol_results import SymbolResults
import logging


@dataclass
class StrategyBacktestReport:
    """Multi-symbol backtest report with aggregated metrics and per-symbol analysis"""

    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float
    symbol_results: Dict[str, SymbolResults]

    def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate aggregated portfolio metrics"""
        # Convert total returns from percentage strings to floats
        total_returns = []
        for results in self.symbol_results.values():
            try:
                # Extract total return value and convert from percentage string
                return_str = results.metrics["Performance Metrics"][1]["value"].strip(
                    "%"
                )  # Using Total Return metric
                total_return = float(return_str) / 100  # Convert percentage to decimal
                total_returns.append(total_return)
            except (ValueError, KeyError, IndexError) as e:
                logging.warning(f"Error processing total return: {e}")
                continue

        # Calculate Sharpe ratios
        sharpe_ratios = []
        for results in self.symbol_results.values():
            try:
                sharpe_str = results.metrics["Performance Metrics"][3][
                    "value"
                ]  # Using Sharpe Ratio metric
                sharpe_ratios.append(float(sharpe_str))
            except (ValueError, KeyError, IndexError) as e:
                logging.warning(f"Error processing Sharpe ratio: {e}")
                continue

        # Calculate drawdowns
        drawdowns = []
        for results in self.symbol_results.values():
            try:
                drawdown_str = results.metrics["Risk Metrics"][0]["value"].strip(
                    "%"
                )  # Using Max Drawdown metric
                drawdowns.append(float(drawdown_str))
            except (ValueError, KeyError, IndexError) as e:
                logging.warning(f"Error processing drawdown: {e}")
                continue

        # Calculate portfolio-level metrics
        avg_total_return = np.mean(total_returns) if total_returns else 0
        avg_sharpe_ratio = np.mean(sharpe_ratios) if sharpe_ratios else 0
        max_drawdown = min(drawdowns) if drawdowns else 0

        return {
            "total_return": avg_total_return * 100,  # Convert back to percentage
            "sharpe_ratio": avg_sharpe_ratio,
            "max_drawdown": max_drawdown,
            "correlation_matrix": self._calculate_correlation_matrix(),
        }

    def _calculate_correlation_matrix(
        self, method: str = "pearson", min_periods: int = 30
    ) -> pd.DataFrame:
        """Calculate correlation matrix between symbols using log returns

        Args:
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            min_periods: Minimum number of overlapping periods required

        Returns:
            DataFrame containing the correlation matrix
        """
        try:
            # Create DataFrame with equity curves for all symbols
            equity_curves = pd.DataFrame(
                {
                    symbol: results.equity_curve
                    for symbol, results in self.symbol_results.items()
                }
            )

            # Calculate log returns
            log_returns = np.log(equity_curves / equity_curves.shift(1)).dropna()

            # Calculate correlation matrix with minimum periods requirement
            correlation_matrix = log_returns.corr(
                method=method, min_periods=min_periods
            )

            # Replace any remaining NaN values with 0
            correlation_matrix = correlation_matrix.fillna(0)

            return correlation_matrix

        except Exception as e:
            logging.error(f"Error calculating correlation matrix: {e}")
            # Return empty correlation matrix in case of error
            symbols = list(self.symbol_results.keys())
            return pd.DataFrame(0, index=symbols, columns=symbols)

    def _prepare_correlation_data(self) -> List[List]:
        """Prepare correlation data for heatmap visualization"""
        corr_matrix = self._calculate_correlation_matrix()
        correlation_data = []

        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                correlation_data.append(
                    [
                        i,  # x
                        j,  # y
                        round(corr_matrix.iloc[i, j], 2),  # correlation value
                    ]
                )

        return correlation_data

    def _prepare_portfolio_equity_data(self) -> List[List]:
        """Prepare combined portfolio equity curve data"""
        # Combine equity curves with equal weighting
        combined_equity = pd.DataFrame(
            {
                symbol: results.equity_curve
                for symbol, results in self.symbol_results.items()
            }
        ).mean(axis=1)

        return [
            [int(t.timestamp() * 1000), float(v)] for t, v in combined_equity.items()
        ]

    def _prepare_symbol_price_data(
        self, symbol: str, results: SymbolResults
    ) -> List[List]:
        """Prepare price data for symbol chart"""
        if not hasattr(results, "portfolio") or results.portfolio is None:
            logging.warning(f"No portfolio data available for symbol {symbol}")
            return []

        return [
            [int(t.timestamp() * 1000), float(p)]
            for t, p in zip(results.portfolio.index, results.portfolio["close"])
        ]

    def _prepare_symbol_trade_annotations(self, trades: List[Dict]) -> List[Dict]:
        """Prepare trade annotations for symbol chart"""
        annotations = []
        for trade in trades:
            timestamp = pd.Timestamp(trade["timestamp"]).timestamp() * 1000
            price = float(trade["price"])
            trade_type = trade["type"].upper()

            annotation = {
                "x": int(timestamp),
                "y": price,
                "title": "↑" if trade_type == "BUY" else "↓",
                "text": f"{trade_type} @ ${price:,.2f}\nQty: {trade['quantity']}",
                "className": f"trade-{trade_type.lower()}",
            }
            annotations.append(annotation)
        return annotations

    def generate_report(self, output_dir: str) -> str:
        """Generate HTML report for multi-symbol backtest"""
        template_path = os.path.join(
            os.path.dirname(__file__), "../templates/strategy_backtest_report.html"
        )
        with open(template_path, "r",encoding="utf-8") as f:
            template = Template(f.read())

        # Calculate portfolio-level metrics
        portfolio_metrics = self._calculate_portfolio_metrics()

        # Prepare data for template
        correlation_symbols = list(self.symbol_results.keys())
        correlation_data = self._prepare_correlation_data()
        portfolio_equity_data = self._prepare_portfolio_equity_data()
        #print(portfolio_equity_data[:5])


        # Format metrics for display with proper extraction
        overall_metrics = [
            {
                "title": "Total Return",
                "value": f"{portfolio_metrics['total_return']:.2f}%",
                "color": (
                    "positive" if portfolio_metrics["total_return"] > 0 else "negative"
                ),
            },
            {
                "title": "Sharpe Ratio",
                "value": f"{portfolio_metrics['sharpe_ratio']:.2f}",
            },
            {
                "title": "Max Drawdown",
                "value": f"{portfolio_metrics['max_drawdown']:.2f}%",
                "color": "negative",
            },
        ]

        # Convert numpy types to native Python types in symbol results
        symbol_results_data = {}
        for symbol, results in self.symbol_results.items():
            # Extract metrics properly
            perf_metrics = {
                m["title"]: m["value"] for m in results.metrics["Performance Metrics"]
            }
            risk_metrics = {
                m["title"]: m["value"] for m in results.metrics["Risk Metrics"]
            }
            trade_metrics = {
                m["title"]: m["value"] for m in results.metrics["Trading Statistics"]
            }

            timestamps = [
                int(pd.Timestamp(t).timestamp() * 1000)
                for t in results.equity_curve.index
            ]
            bench = results.benchmark_data["value"].reindex(results.equity_curve.index).ffill()

            symbol_results_data[symbol] = {
                "trades": [
                    {
                        "timestamp": (
                            pd.Timestamp(trade["timestamp"]).strftime("%Y-%m-%d %H:%M")
                            if isinstance(trade["timestamp"], str)
                            else trade["timestamp"].strftime("%Y-%m-%d %H:%M")
                        ),
                        "type": trade["type"].upper(),
                        "price": float(trade["price"]),
                        "quantity": int(trade["quantity"]),
                        "cost": float(abs(trade.get("cost", 0))),
                        "commission": float(trade.get("commission", 0)),
                        "pnl": float(trade.get("pnl", 0)),
                    }
                    for trade in results.trades
                ],
                "metrics": {
                    "Performance Metrics": [
                        {
                            "title": "Total Return",
                            "value": perf_metrics["Total Return"],
                            "color": (
                                "positive"
                                if float(perf_metrics["Total Return"].strip("%")) > 0
                                else "negative"
                            ),
                        },
                        {
                            "title": "Annual Return",
                            "value": perf_metrics["Annual Return"],
                            "color": (
                                "positive"
                                if float(perf_metrics["Annual Return"].strip("%")) > 0
                                else "negative"
                            ),
                        },
                        {
                            "title": "Sharpe Ratio",
                            "value": perf_metrics["Sharpe Ratio"],
                        },
                        {
                            "title": "Sortino Ratio",
                            "value": perf_metrics["Sortino Ratio"],
                        },
                    ],
                    "Risk Metrics": [
                        {
                            "title": "Max Drawdown",
                            "value": risk_metrics["Max Drawdown"],
                            "color": "negative",
                        },
                        {"title": "Volatility", "value": risk_metrics["Volatility"]},
                        {
                            "title": "Value at Risk",
                            "value": risk_metrics["Value at Risk (95%)"],
                        },
                    ],
                    "Trading Statistics": [
                        {
                            "title": "Total Trades",
                            "value": trade_metrics["Total Trades"],
                        },
                        {"title": "Win Rate", "value": trade_metrics["Win Rate"]},
                        {
                            "title": "Profit Factor",
                            "value": trade_metrics["Profit Factor"],
                        },
                        {
                            "title": "Average Trade Return",
                            "value": trade_metrics["Average Trade Return"],
                            "color": (
                                "positive"
                                if float(
                                    trade_metrics["Average Trade Return"]
                                    .strip("$")
                                    .replace(",", "")
                                )
                                > 0
                                else "negative"
                            ),
                        },
                        {
                            "title": "Average Win",
                            "value": trade_metrics["Average Win"],
                            "color": "positive",
                        },
                        {
                            "title": "Average Loss",
                            "value": trade_metrics["Average Loss"],
                            "color": "negative",
                        },
                        {
                            "title": "Largest Win",
                            "value": trade_metrics["Largest Win"],
                            "color": "positive",
                        },
                        {
                            "title": "Largest Loss",
                            "value": trade_metrics["Largest Loss"],
                            "color": "negative",
                        },
                        {
                            "title": "Max Consec. Wins",
                            "value": trade_metrics["Max Consecutive Wins"],
                        },
                        # {'title': 'Max Consecutive Losses', 'value': trade_metrics['Max Consecutive Losses']}
                    ],
                },
                "total_return": float(perf_metrics["Total Return"].strip("%")),
                "equity_curve": [
                    [ts, float(v)]
                    for ts, v in zip(timestamps, results.equity_curve.values)
                ],
                "benchmark_curve": [
                    [ts, float(v)]
                    for ts, v in zip(timestamps, bench.values)

                ],
                "drawdown": [
                    [int(pd.Timestamp(t).timestamp() * 1000), float(v)]
                    for t, v in results.drawdown.items()
                ],
                "monthly_returns": [
                    [int(pd.Timestamp(t).timestamp() * 1000), float(v)]
                    for t, v in results.monthly_returns["returns"].items()
                ],
                "price_data": self._prepare_symbol_price_data(symbol, results),
                "trade_annotations": self._prepare_symbol_trade_annotations(
                    results.trades
                ),
            }

        # Generate HTML content
        html_content = template.render(
            strategy_name=self.strategy_name,
            start_date=self.start_date.strftime("%Y-%m-%d"),
            end_date=self.end_date.strftime("%Y-%m-%d"),
            overall_metrics=overall_metrics,
            correlation_symbols=correlation_symbols,
            correlation_data=correlation_data,
            portfolio_equity_data=portfolio_equity_data,
            symbol_results=symbol_results_data,
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Write HTML file
        symbols_str = "_".join(self.symbol_results.keys())
        filename = f"{self.strategy_name}_{symbols_str}_backtest_report.html"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

            # 同时建议让方法返回路径，方便调用方：
        return output_path
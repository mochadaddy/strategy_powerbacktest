from typing import Dict
import pandas as pd


class BenchmarkPortfolio:
    def __init__(self, initial_capital: float, lot_size: int):
        self.initial_capital = initial_capital
        self.lot_size = lot_size
        self.history = pd.DataFrame(columns=["timestamp", "value"])

    def calculate_buy_and_hold(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate buy and hold performance using initial capital"""
        data = data.set_index(pd.to_datetime(data["time_key"]))
        initial_price = data["close"].iloc[0]

        # Calculate maximum shares that could be bought with initial capital
        max_shares = int(self.initial_capital / initial_price)
        shares = (max_shares // self.lot_size) * self.lot_size

        if shares < self.lot_size:
            # Not enough capital to buy minimum lot
            portfolio_values = pd.DataFrame(
                {"value": self.initial_capital}, index=data.index
            )
        else:
            initial_cost = shares * initial_price
            remaining_cash = self.initial_capital - initial_cost

            # Calculate portfolio value over time
            portfolio_values = pd.DataFrame(
                {"value": data["close"] * shares + remaining_cash}, index=data.index
            )

        return portfolio_values

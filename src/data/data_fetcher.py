from futu import *
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from src.utils.logger import setup_logger
from src.utils.timeframe import get_kl_type
from src.utils.data_resampler import resample_ohlcv

logger = setup_logger(__name__)


class FutuDataFetcher:
    # FutuDataFetcher在实例化过程中初始化库OpenQuoteContext中的一个实例futu
    def __init__(self, host: str = "localhost", port: int = 11111):
        self.quote_ctx = OpenQuoteContext(host=host, port=port)
        logger.info("Initialized Futu OpenQuoteContext")

    def __del__(self):
        self.quote_ctx.close()

    # 历史数据是通过_fetch_historical_data与Futu request_history_klineAPI交互的方式获取的。
    def _fetch_historical_data(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "DAY"
    ) -> pd.DataFrame:
        """
        Fetch historical data from Futu OpenAPI and resample if needed

        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            timeframe: K-line timeframe (e.g., "1M", "4H", "DAY")

        Returns:
            DataFrame with historical data
        """
        try:
            # Map custom timeframes to available Futu timeframes
            futu_timeframe_map = {
                "4H": "60M",  # Fetch 1-hour data for 4H resampling
                "2H": "60M",  # Fetch 1-hour data for 2H resampling
                "3H": "60M",  # Fetch 1-hour data for 3H resampling
            }

            # Use mapped timeframe for Futu API
            fetch_timeframe = futu_timeframe_map.get(timeframe.upper(), timeframe)
            kl_type = get_kl_type(fetch_timeframe)

            ret_code, data, page_req_key = self.quote_ctx.request_history_kline(
                symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                ktype=kl_type,
            )

            if ret_code != RET_OK:
                logger.error(f"Failed to fetch data: {data}")
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # Resample if needed
            if timeframe.upper() in futu_timeframe_map:
                df = resample_ohlcv(df, timeframe)

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise

    def get_trading_calendar(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DatetimeIndex:
        """
        Get trading calendar between start and end dates using Futu OpenAPI

        Args:
            start_date: Start date for calendar
            end_date: End date for calendar

        Returns:
            DatetimeIndex of trading days
        """
        try:
            # Convert dates to string format required by Futu API
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Request trading days from Futu API
            ret_code, data = self.quote_ctx.request_trading_days(
                market=TrdMarket.HK,  # Hong Kong market
                start_date=start_str,
                end_date=end_str,
            )

            if ret_code != RET_OK:
                self.logger.error(f"Failed to fetch trading calendar: {data}")
                return pd.DatetimeIndex([])

            # Convert to DatetimeIndex and sort
            calendar = pd.DatetimeIndex(data["trading_days"]).sort_values()

            return calendar

        except Exception as e:
            logger.error(f"Error getting trading calendar: {str(e)}")
            return pd.DatetimeIndex([])

    def fetch_lot_size(self, symbol: str) -> int:
        """
        Fetch lot size for a given symbol from Futu API

        Args:
            symbol: Stock symbol

        Returns:
            int: Lot size for the symbol
        """
        try:
            ret_code, data = self.quote_ctx.get_market_snapshot([symbol])
            if ret_code != RET_OK:
                logger.error(f"Failed to fetch lot size for {symbol}: {data}")
                return 1  # Default lot size

            return int(data["lot_size"][0])

        except Exception as e:
            logger.error(f"Error fetching lot size for {symbol}: {str(e)}")
            return 1  # Default lot size

    def fetch_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fundamental data for a given symbol from Futu API

        Args:
            symbol: Stock symbol

        Returns:
            Dict containing fundamental data
        """
        try:
            # Fetch basic information
            ret_code, snapshot_data = self.quote_ctx.get_market_snapshot([symbol])
            if ret_code != RET_OK:
                logger.error(
                    f"Failed to fetch market snapshot for {symbol}: {snapshot_data}"
                )
                return {}

            # Fetch company information
            ret_code, company_data = self.quote_ctx.get_stock_basicinfo(
                Market.HK, SecurityType.STOCK, [symbol]
            )
            if ret_code != RET_OK:
                logger.error(
                    f"Failed to fetch company data for {symbol}: {company_data}"
                )
                return {}

            # Extract and combine data
            fundamental_data = {
                "market_cap": float(snapshot_data["market_cap"][0]),
                "pe_ratio": float(snapshot_data["pe_ratio"][0]),
                "pb_ratio": float(snapshot_data["pb_ratio"][0]),
                "dividend_yield": float(snapshot_data["dividend_ratio"][0]),
                "lot_size": int(snapshot_data["lot_size"][0]),
                "stock_name": company_data["name"][0],
                "stock_type": company_data["stock_type"][0],
                "listing_date": company_data["listing_date"][0],
                "total_shares": float(company_data["total_shares"][0]),
                "float_shares": float(company_data["float_shares"][0]),
                "current_price": float(snapshot_data["last_price"][0]),
                "fifty_two_week_high": float(snapshot_data["high_price"][0]),
                "fifty_two_week_low": float(snapshot_data["low_price"][0]),
                "avg_volume_3m": float(snapshot_data["volume_3m"][0]),
                "industry": company_data["industry"][0],
            }

            return fundamental_data

        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {}

    def fetch_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        warmup_periods: int = 0,
    ) -> pd.DataFrame:
        """
        Fetch data with warmup period included, always requesting extra 100 days

        Args:
            symbol: Stock symbol
            start_date: Start date for trading period
            end_date: End date for trading period
            timeframe: Data timeframe
            warmup_periods: Number of additional periods needed before start_date (ignored)

        Returns:
            DataFrame with warmup period included
        """
        # Always request 100 extra days of data
        adjusted_start = start_date - pd.Timedelta(days=100)

        # Fetch data including warmup period
        data = self._fetch_historical_data(symbol, adjusted_start, end_date, timeframe)

        if data is None or data.empty:
            raise ValueError(f"No data available for {symbol}")

        return data

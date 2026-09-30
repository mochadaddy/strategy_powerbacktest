import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from typing import Dict, Optional
import sqlite3
import json
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

DATE_FORMAT = "%Y-%m-%d"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

class DataStore:
    """
    Handles storage and retrieval of historical market data.
    Supports both in-memory and persistent storage.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.in_memory_store: Dict[str, pd.DataFrame] = {}

        if db_path:
            self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize SQLite database with required tables"""
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS market_data (
                        symbol TEXT,
                        timestamp TEXT,
                        data JSON,
                        interval TEXT,
                        PRIMARY KEY (symbol, timestamp, interval)
                    )
                """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS data_coverage (
                        symbol TEXT,
                        interval TEXT,
                        start_date TEXT,
                        end_date TEXT,
                        PRIMARY KEY (symbol, interval, start_date)
                    )
                """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS symbol_info (
                        symbol TEXT PRIMARY KEY,
                        lot_size INTEGER
                    )
                """
                )
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def save_data(
        self, symbol: str, data: pd.DataFrame, interval: str, persistent: bool = True
    ) -> None:
        """
        Save market data to storage.

        Args:
            symbol: Market symbol
            data: DataFrame containing market data
            interval: Time interval of the data
            persistent: Whether to save to persistent storage
        """
        # Save to in-memory store
        key = f"{symbol}_{interval}"
        self.in_memory_store[key] = data

        # Save to persistent storage if enabled
        if persistent and self.db_path:
            try:
                rows = []
                for idx, record in zip(data.index, data.to_dict("records")):
                    ts = record["time_key"] if "time_key" in record else idx
                    rows.append(
                        (
                            symbol,
                            pd.to_datetime(ts).isoformat(),
                            json.dumps(record, default=_json_default),
                            interval,)
                    )
                with sqlite3.connect(self.db_path) as conn:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO market_data
                        (symbol, timestamp, data, interval)
                        VALUES (?, ?, ?, ?)
                        """,
                        rows,
                    )

            except Exception as e:
                logger.error(f"Failed to save data to database: {str(e)}")
                raise

    def load_data(
        self, symbol: str, interval: str, use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load market data from storage.

        Args:
            symbol: Market symbol
            interval: Time interval of the data
            use_cache: Whether to use in-memory cache

        Returns:
            DataFrame containing market data or None if not found
        """
        key = f"{symbol}_{interval}"

        # Try in-memory store first
        if use_cache and key in self.in_memory_store:
            return self.in_memory_store[key]

        # Try persistent storage
        if self.db_path:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    query = """
                        SELECT timestamp, data
                        FROM market_data
                        WHERE symbol = ? AND interval = ?
                        ORDER BY timestamp
                    """
                    rows = conn.execute(query, (symbol, interval)).fetchall()

                    if not rows:
                        return None

                    data = []
                    for timestamp, json_data in rows:
                        row_data = json.loads(json_data)
                        data.append(row_data)

                    df = pd.DataFrame(data)
                    df.index = pd.to_datetime([row[0] for row in rows])

                    # Update cache
                    self.in_memory_store[key] = df
                    return df

            except Exception as e:
                logger.error(f"Failed to load data from database: {str(e)}")
                raise

        return None

    def load_range(
            self, symbol: str, interval: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Load cached market data for [start_date, end_date] (dates inclusive).

        Returns None unless the whole range was previously fetched and recorded
        via `mark_covered`, so a partial cache never masquerades as complete data.
        The returned frame has a RangeIndex and a `time_key` column, matching the
        format returned by the data fetcher.
        """
        if not self.db_path or not self.is_covered(
                symbol, interval, start_date, end_date
        ):
            return None

        range_start = pd.Timestamp(start_date).normalize().isoformat()
        range_end = (
                pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
        ).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT data
                FROM market_data
                WHERE symbol = ? AND interval = ? AND timestamp >= ? AND timestamp < ?
                ORDER BY timestamp
                """,
                (symbol, interval, range_start, range_end),
            ).fetchall()

        if not rows:
            return None
        return pd.DataFrame([json.loads(row[0]) for row in rows])

    def _get_coverage(self, symbol: str, interval: str) -> List[Tuple[str, str]]:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                """
                SELECT start_date, end_date
                FROM data_coverage
                WHERE symbol = ? AND interval = ?
                ORDER BY start_date
                """,
                (symbol, interval),
            ).fetchall()

    def is_covered(
            self, symbol: str, interval: str, start_date: datetime, end_date: datetime
    ) -> bool:
        """Check whether [start_date, end_date] lies within a fetched range."""
        if not self.db_path:
            return False
        start = pd.Timestamp(start_date).strftime(DATE_FORMAT)
        end = pd.Timestamp(end_date).strftime(DATE_FORMAT)
        return any(
            cov_start <= start and cov_end >= end
            for cov_start, cov_end in self._get_coverage(symbol, interval)
        )

    def mark_covered(
            self, symbol: str, interval: str, start_date: datetime, end_date: datetime
    ) -> None:
        """
        Record that data for [start_date, end_date] has been fully fetched.

        The range is capped at yesterday because today's bars are still forming,
        and overlapping or adjacent ranges are merged.
        """
        if not self.db_path:
            return
        start = pd.Timestamp(start_date).normalize()
        end = min(
            pd.Timestamp(end_date).normalize(),
            pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
        )
        if end < start:
            return

        ranges = [
            (pd.Timestamp(s), pd.Timestamp(e))
            for s, e in self._get_coverage(symbol, interval)
        ]
        ranges.append((start, end))
        ranges.sort()

        merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for range_start, range_end in ranges:
            if merged and range_start <= merged[-1][1] + timedelta(days=1):
                merged[-1] = (merged[-1][0], max(merged[-1][1], range_end))
            else:
                merged.append((range_start, range_end))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM data_coverage WHERE symbol = ? AND interval = ?",
                (symbol, interval),
            )
            conn.executemany(
                """
                INSERT INTO data_coverage (symbol, interval, start_date, end_date)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        symbol,
                        interval,
                        s.strftime(DATE_FORMAT),
                        e.strftime(DATE_FORMAT),
                    )
                    for s, e in merged
                ],
            )

    def load_lot_size(self, symbol: str) -> Optional[int]:
        """Load cached lot size for a symbol"""
        if not self.db_path:
            return None
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT lot_size FROM symbol_info WHERE symbol = ?", (symbol,)
            ).fetchone()
        return int(row[0]) if row else None

    def save_lot_size(self, symbol: str, lot_size: int) -> None:
        """Cache lot size for a symbol"""
        if not self.db_path:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO symbol_info (symbol, lot_size) VALUES (?, ?)",
                (symbol, int(lot_size)),
            )
    def clear_cache(self) -> None:
        """Clear the in-memory cache"""
        self.in_memory_store.clear()

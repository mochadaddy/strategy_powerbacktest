import pandas as pd
from typing import Dict, Optional
import sqlite3
import json
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


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
                with sqlite3.connect(self.db_path) as conn:
                    for _, row in data.iterrows():
                        ts = row["time_key"] if "time_key" in row.index else row.name
                        ts = pd.to_datetime(ts).isoformat()
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO market_data
                            (symbol, timestamp, data, interval)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                symbol,
                                ts,
                                #row.name.isoformat(),
                                json.dumps(row.to_dict()),
                                interval,
                            ),
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

    def clear_cache(self) -> None:
        """Clear the in-memory cache"""
        self.in_memory_store.clear()

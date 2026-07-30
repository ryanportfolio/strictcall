"""Backend selection: one env var picks the warehouse, tools never branch on it."""

import os

from strictcall.backends.base import BackendError, SqlBackend
from strictcall.backends.duckdb_ import DuckDBBackend
from strictcall.backends.snowflake_ import SnowflakeBackend

__all__ = ["BackendError", "DuckDBBackend", "SnowflakeBackend", "SqlBackend", "get_backend"]

DEFAULT_DB_PATH = "data/loyalty.duckdb"


def get_backend(database: str | None = None) -> SqlBackend:
    kind = os.environ.get("STRICTCALL_BACKEND", "duckdb")
    if kind == "duckdb":
        return DuckDBBackend(database or os.environ.get("STRICTCALL_DB", DEFAULT_DB_PATH))
    if kind == "snowflake":
        return SnowflakeBackend()
    raise ValueError(f"Unknown STRICTCALL_BACKEND={kind!r}; expected 'duckdb' or 'snowflake'.")

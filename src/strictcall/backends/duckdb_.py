"""DuckDB implementation of the SqlBackend contract.

Keeps the repository runnable end-to-end with no credentials and no cloud spend.
"""

import datetime
import threading
from decimal import Decimal
from pathlib import Path
from time import perf_counter

import duckdb

from strictcall.backends.base import BackendError, ensure_single_select
from strictcall.contracts import (
    ColumnInfo,
    Scalar,
    SchemaDescription,
    SqlQueryResult,
    TableInfo,
)

_SCHEMA_SQL = """
SELECT table_name, column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'main'
ORDER BY table_name, ordinal_position
"""


def _to_scalar(value: object) -> Scalar:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime.datetime | datetime.date | datetime.time):
        return value.isoformat()
    return str(value)


class DuckDBBackend:
    name = "duckdb"

    def __init__(
        self,
        database: str | Path = ":memory:",
        connection: duckdb.DuckDBPyConnection | None = None,
        timeout_s: float = 15.0,
    ) -> None:
        if connection is not None:
            self._con = connection
        else:
            database = Path(database) if database != ":memory:" else database
            if isinstance(database, Path) and not database.exists():
                raise BackendError(
                    f"Database file {database} does not exist. "
                    "Generate it with: python -m strictcall.dataset generate"
                )
            self._con = duckdb.connect(str(database), read_only=isinstance(database, Path))
        self._timeout_s = timeout_s

    def execute(self, query: str, limit: int) -> SqlQueryResult:
        clean = ensure_single_select(query)
        wrapped = f"SELECT * FROM ({clean}) AS strictcall_q LIMIT {limit + 1}"
        watchdog = threading.Timer(self._timeout_s, self._con.interrupt)
        watchdog.start()
        start = perf_counter()
        try:
            cursor = self._con.execute(wrapped)
            columns = [desc[0] for desc in cursor.description]
            raw_rows = cursor.fetchall()
        except duckdb.InterruptException as exc:
            raise BackendError(f"Query exceeded the {self._timeout_s}s timeout.") from exc
        except duckdb.Error as exc:
            raise BackendError(str(exc)) from exc
        finally:
            watchdog.cancel()
        elapsed_ms = (perf_counter() - start) * 1000
        truncated = len(raw_rows) > limit
        rows = [[_to_scalar(v) for v in row] for row in raw_rows[:limit]]
        return SqlQueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            backend=self.name,
            elapsed_ms=round(elapsed_ms, 2),
        )

    def describe(self) -> SchemaDescription:
        rows = self._con.execute(_SCHEMA_SQL).fetchall()
        tables: dict[str, list[ColumnInfo]] = {}
        for table_name, column_name, data_type, is_nullable in rows:
            tables.setdefault(table_name, []).append(
                ColumnInfo(name=column_name, type=data_type, nullable=is_nullable == "YES")
            )
        return SchemaDescription(
            backend=self.name,
            tables=[TableInfo(name=name, columns=cols) for name, cols in sorted(tables.items())],
        )

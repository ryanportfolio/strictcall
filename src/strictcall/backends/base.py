"""The single SQL backend interface both DuckDB and Snowflake implement."""

from typing import Protocol, runtime_checkable

import duckdb

from strictcall.contracts import BackendName, SchemaDescription, SqlQueryResult


class BackendError(Exception):
    """Raised when a query is rejected before or during execution."""


@runtime_checkable
class SqlBackend(Protocol):
    """Contract every SQL backend satisfies. Tools depend on this, never on a
    concrete backend."""

    name: BackendName

    def execute(self, query: str, limit: int) -> SqlQueryResult: ...

    def describe(self) -> SchemaDescription: ...


def ensure_single_select(query: str) -> str:
    """Parse the query and reject anything that is not exactly one SELECT.

    Uses DuckDB's parser rather than string matching, so comments, casing, and
    embedded keywords cannot smuggle DDL/DML through. Returns the query with any
    trailing semicolon stripped, ready to be wrapped in a LIMIT subquery.
    """
    try:
        statements = duckdb.extract_statements(query)
    except duckdb.Error as exc:
        raise BackendError(f"Could not parse SQL: {exc}") from exc
    if len(statements) != 1:
        raise BackendError("Exactly one SQL statement is allowed per call.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise BackendError("Only SELECT statements are allowed; DDL and DML are rejected.")
    return query.strip().rstrip(";")

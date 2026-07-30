"""SQL tools: contract-validated query execution and schema introspection.

Tool failures never raise - they return a serialized ToolError so the model
sees what went wrong and can correct its next call. That message loop is the
tool-level retry mechanism.
"""

from langchain_core.tools import StructuredTool

from strictcall.backends.base import BackendError, SqlBackend
from strictcall.contracts import SqlQueryInput, ToolError

SQL_QUERY_DESCRIPTION = (
    "Run a single read-only SELECT statement against the loyalty warehouse. "
    "Returns JSON with columns, rows, row_count, truncated, backend, elapsed_ms. "
    "If truncated is true, the result was capped at the requested limit."
)

DESCRIBE_SCHEMA_DESCRIPTION = (
    "List every table and view in the loyalty warehouse with column names, "
    "types, and nullability. Call this before writing SQL against unfamiliar tables."
)


def make_sql_tools(backend: SqlBackend) -> list[StructuredTool]:
    def run_sql_query(query: str, limit: int = 50) -> str:
        try:
            return backend.execute(query, limit).model_dump_json()
        except BackendError as exc:
            return ToolError(
                error=str(exc),
                hint="Submit exactly one SELECT statement. Use describe_schema to "
                "check table and column names.",
            ).model_dump_json()

    def run_describe_schema() -> str:
        return backend.describe().model_dump_json()

    return [
        StructuredTool.from_function(
            func=run_sql_query,
            name="sql_query",
            description=SQL_QUERY_DESCRIPTION,
            args_schema=SqlQueryInput,
        ),
        StructuredTool.from_function(
            func=run_describe_schema,
            name="describe_schema",
            description=DESCRIBE_SCHEMA_DESCRIPTION,
        ),
    ]

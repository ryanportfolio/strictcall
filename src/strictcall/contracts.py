"""Pydantic contracts for every tool input and output.

Each tool the agent can call validates its arguments against an input model and
serializes its return value from an output model. Nothing crosses the tool
boundary unvalidated, in either direction.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Scalar = str | int | float | bool | None

BackendName = Literal["duckdb", "snowflake"]


class SqlQueryInput(BaseModel):
    """Arguments the model must supply to run a query."""

    query: str = Field(
        description="A single read-only SELECT statement to run against the loyalty warehouse."
    )
    limit: int = Field(
        default=50, ge=1, le=500, description="Maximum number of rows to return (1-500)."
    )


class SqlQueryResult(BaseModel):
    """What the sql_query tool guarantees back on success."""

    columns: list[str]
    rows: list[list[Scalar]]
    row_count: int
    truncated: bool
    backend: BackendName
    elapsed_ms: float


class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool


class TableInfo(BaseModel):
    name: str
    columns: list[ColumnInfo]


class SchemaDescription(BaseModel):
    """What the describe_schema tool guarantees back."""

    backend: BackendName
    tables: list[TableInfo]


class FxRateInput(BaseModel):
    """Arguments for the external exchange-rate API tool."""

    base: str = Field(
        pattern=r"^[A-Z]{3}$", description="ISO 4217 currency code to convert from, e.g. USD."
    )
    target: str = Field(
        pattern=r"^[A-Z]{3}$", description="ISO 4217 currency code to convert to, e.g. EUR."
    )


class FxRateResult(BaseModel):
    base: str
    target: str
    rate: float
    as_of: date


class ToolError(BaseModel):
    """Structured error returned to the model so it can self-correct and retry."""

    error: str
    hint: str | None = None


class ToolFailure(BaseModel):
    """A tool call that was rejected, and the correction handed back to the
    model. The turn can still succeed: these are the attempts behind the
    answer, not errors the caller has to handle."""

    tool: str
    error: str
    hint: str | None = None


class Answer(BaseModel):
    """Structured summary of a completed agent turn."""

    text: str
    sql_used: list[str]
    data_caveats: list[str]
    tool_errors: list[ToolFailure] = []

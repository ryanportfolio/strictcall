import httpx
import pytest

from strictcall.contracts import FxRateResult, SchemaDescription, SqlQueryResult, ToolError
from strictcall.tools import make_fx_tool, make_sql_tools


@pytest.fixture
def sql_tools(backend):
    query_tool, describe_tool = make_sql_tools(backend)
    return query_tool, describe_tool


def test_sql_tool_returns_contract_json(sql_tools):
    query_tool, _ = sql_tools
    output = query_tool.invoke({"query": "SELECT COUNT(*) AS n FROM members", "limit": 5})
    result = SqlQueryResult.model_validate_json(output)
    assert result.rows == [[60]]


def test_sql_tool_errors_are_structured_not_raised(sql_tools):
    query_tool, _ = sql_tools
    output = query_tool.invoke({"query": "DROP TABLE members", "limit": 5})
    error = ToolError.model_validate_json(output)
    assert "SELECT" in error.error
    assert error.hint is not None


def test_sql_tool_rejects_out_of_range_limit(sql_tools):
    query_tool, _ = sql_tools
    with pytest.raises(Exception):  # noqa: B017 - pydantic validation propagates via langchain
        query_tool.invoke({"query": "SELECT 1", "limit": 10_000})


def test_describe_tool_returns_contract_json(sql_tools):
    _, describe_tool = sql_tools
    output = describe_tool.invoke({})
    description = SchemaDescription.model_validate_json(output)
    assert description.backend == "duckdb"


def _fx_tool(handler):
    return make_fx_tool(httpx.Client(transport=httpx.MockTransport(handler)))


def test_fx_tool_validates_api_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["base"] == "USD"
        return httpx.Response(
            200, json={"base": "USD", "date": "2026-07-29", "rates": {"EUR": 0.86}}
        )

    output = _fx_tool(handler).invoke({"base": "USD", "target": "EUR"})
    result = FxRateResult.model_validate_json(output)
    assert result.rate == 0.86
    assert result.as_of.isoformat() == "2026-07-29"


def test_fx_tool_turns_http_failures_into_tool_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    output = _fx_tool(handler).invoke({"base": "USD", "target": "EUR"})
    error = ToolError.model_validate_json(output)
    assert "HTTPStatusError" in error.error


def test_fx_tool_rejects_same_currency():
    output = _fx_tool(lambda request: httpx.Response(200)).invoke({"base": "USD", "target": "USD"})
    assert "same currency" in ToolError.model_validate_json(output).error

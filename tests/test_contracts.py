import pytest
from pydantic import ValidationError

from strictcall.contracts import FxRateInput, SqlQueryInput, SqlQueryResult


def test_limit_bounds_are_enforced():
    with pytest.raises(ValidationError):
        SqlQueryInput(query="SELECT 1", limit=0)
    with pytest.raises(ValidationError):
        SqlQueryInput(query="SELECT 1", limit=501)
    assert SqlQueryInput(query="SELECT 1").limit == 50


def test_fx_input_requires_iso_codes():
    with pytest.raises(ValidationError):
        FxRateInput(base="usd", target="EUR")
    with pytest.raises(ValidationError):
        FxRateInput(base="US", target="EUR")
    assert FxRateInput(base="USD", target="EUR").target == "EUR"


def test_sql_result_round_trips_through_json():
    result = SqlQueryResult(
        columns=["n", "when"],
        rows=[[1, "2026-01-01"], [None, "2026-01-02"]],
        row_count=2,
        truncated=False,
        backend="duckdb",
        elapsed_ms=1.5,
    )
    assert SqlQueryResult.model_validate_json(result.model_dump_json()) == result


def test_unknown_backend_name_is_rejected():
    with pytest.raises(ValidationError):
        SqlQueryResult(
            columns=[], rows=[], row_count=0, truncated=False, backend="postgres", elapsed_ms=0
        )

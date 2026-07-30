import pytest

from strictcall.backends.base import BackendError
from strictcall.backends.snowflake_ import SnowflakeBackend


@pytest.mark.parametrize(
    "bad_query",
    [
        "DROP TABLE members",
        "INSERT INTO members VALUES (1, 'x', 'x', 'x', '2026-01-01', 1)",
        "UPDATE members SET city = 'Nowhere'",
        "SELECT 1; SELECT 2",
        "CREATE TABLE evil (id INTEGER)",
        "not sql at all (",
    ],
)
def test_non_select_statements_are_rejected(backend, bad_query):
    with pytest.raises(BackendError):
        backend.execute(bad_query, limit=10)


def test_limit_caps_rows_and_flags_truncation(backend):
    result = backend.execute("SELECT * FROM members", limit=10)
    assert result.row_count == 10
    assert len(result.rows) == 10
    assert result.truncated is True
    assert result.backend == "duckdb"


def test_small_results_are_not_truncated(backend):
    result = backend.execute("SELECT COUNT(*) AS n FROM members", limit=10)
    assert result.truncated is False
    assert result.rows[0][0] == 60


def test_trailing_semicolon_is_tolerated(backend):
    result = backend.execute("SELECT 1 AS one;", limit=5)
    assert result.rows == [[1]]


def test_values_are_json_safe_scalars(backend):
    result = backend.execute(
        "SELECT joined_at, amount_usd FROM members JOIN transactions USING (member_id)", limit=1
    )
    date_value, amount = result.rows[0]
    assert isinstance(date_value, str)  # DATE -> ISO string
    assert isinstance(amount, float)  # DECIMAL -> float


def test_bad_column_becomes_backend_error(backend):
    with pytest.raises(BackendError):
        backend.execute("SELECT nonexistent_column FROM members", limit=5)


def test_describe_lists_all_tables_and_the_view(backend):
    description = backend.describe()
    names = {table.name for table in description.tables}
    assert {"tiers", "members", "transactions", "redemptions", "v_balances"} <= names
    members = next(t for t in description.tables if t.name == "members")
    assert {c.name for c in members.columns} == {
        "member_id",
        "full_name",
        "email",
        "city",
        "joined_at",
        "tier_id",
    }


def test_snowflake_backend_is_a_phase_3_stub():
    with pytest.raises(NotImplementedError):
        SnowflakeBackend()

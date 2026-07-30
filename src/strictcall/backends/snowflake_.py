"""Snowflake implementation of the SqlBackend contract (Phase 3).

The class exists so backend dispatch is real today; the implementation lands
with the Snowflake trial account in Phase 3. It will satisfy the same
SqlBackend protocol and reuse ensure_single_select from base.
"""


class SnowflakeBackend:
    name = "snowflake"

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "SnowflakeBackend arrives in Phase 3. "
            "Set STRICTCALL_BACKEND=duckdb (the default) to run without credentials."
        )

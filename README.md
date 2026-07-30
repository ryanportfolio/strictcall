# strictcall

A LangGraph agent that answers natural-language questions against a SQL warehouse.
Every tool call is validated against an explicit Pydantic contract — on the way in
*and* on the way out. One SQL interface, two backends: DuckDB (runs anywhere, zero
credentials) and Snowflake (same contract, cloud warehouse). Built to package for
Amazon Bedrock AgentCore Runtime.

## Run it locally

```bash
uv sync
uv run python -m strictcall.dataset generate   # builds the demo warehouse, no data files in git
export ANTHROPIC_API_KEY=...                   # any Claude model; override with STRICTCALL_MODEL
uv run strictcall "Which members are within 500 points of the next tier?"
```

Tests and lint need no API key and no network:

```bash
uv run pytest && uv run ruff check .
```

## Example

<!-- EXAMPLE_OUTPUT -->

## Why every tool call is schema-validated

LLMs emit JSON; databases execute strings. The gap between those two is where
agent bugs live. strictcall closes it from both directions:

- **Inputs** — each tool's arguments are a Pydantic model
  ([contracts.py](src/strictcall/contracts.py)). `limit` outside 1–500 or a
  malformed currency code is rejected before any code runs; the model gets the
  validation error back and corrects itself.
- **Execution guards** — the SQL tool parses the statement (DuckDB's parser, not
  regex) and rejects anything that is not exactly one `SELECT`. A wrapping
  `LIMIT` caps rows server-side, and a watchdog interrupts runaway queries.
- **Outputs** — results are validated into `SqlQueryResult` / `FxRateResult`
  before serialization, so the model always sees the same shape: columns, rows,
  `truncated`, backend, timing. Failures come back as a structured `ToolError`
  with a hint, which is what makes the agent's retry loop reliable rather than
  accidental.

## Architecture

```
question ──> agent node (Claude, RetryPolicy) ──> tool node ──> agent node ──> answer
                    │                                │
             conversation memory              Pydantic contracts
             (checkpointer, thread_id)     ┌─────────┴─────────┐
                                        sql_query          fx_rate
                                        describe_schema   (Frankfurter API)
                                           │
                                     SqlBackend protocol
                                     ┌─────┴─────────────┐
                                  DuckDB            Snowflake
```

- Hand-built `StateGraph` (not a prebuilt agent constructor): model node with a
  `RetryPolicy` for transient API failures, conditional edge to a tool node,
  loop until the model stops calling tools.
- Token streaming via `stream_mode="messages"`; conversation memory via a
  checkpointer keyed by `--thread`.
- The tool layer holds one `SqlBackend` reference and never branches on which
  warehouse is behind it — `STRICTCALL_BACKEND=duckdb|snowflake` picks the
  implementation.

## Dataset

Synthetic loyalty program: `tiers`, `members`, `transactions`, `redemptions`,
and a `v_balances` view (lifetime points, balance, points to next tier). The
generator is deterministic — same seed, same database — and deliberately places
a dozen members just under their next tier so the demo questions have real
answers. Nothing is checked in; `python -m strictcall.dataset generate` rebuilds
it in about a second.

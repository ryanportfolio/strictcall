# strictcall

A LangGraph agent that answers plain-English questions against a SQL warehouse.
Every tool call is validated against a Pydantic contract, inputs and outputs
both. One SQL interface, two backends: DuckDB (runs anywhere, no credentials)
and Snowflake (same contract, cloud warehouse). Built to package for Amazon
Bedrock AgentCore Runtime.

## Run it locally

```bash
uv sync
uv run python -m strictcall.dataset generate    # builds the demo warehouse; no data files in git
export ANTHROPIC_API_KEY=...                    # default model is claude-opus-5
uv run strictcall "Which members are within 500 points of the next tier?"
```

No Anthropic key? Any OpenRouter model works, including free ones:

```bash
export OPENROUTER_API_KEY=...
export STRICTCALL_MODEL="openrouter:cohere/north-mini-code:free"
```

Tests and lint need no key and no network:

```bash
uv run pytest && uv run ruff check .
```

## Example

Real output from the run above (`--json` adds the structured answer; member
lists shortened here for length):

```
$ uv run strictcall --json "Which members are within 500 points of reaching the next loyalty tier?"

[describe_schema]
[sql_query]
Here are the members within 500 points of reaching their next loyalty tier (29 total):

**Members Closest to Next Tier (1-50 points):**
- Kira Ekwueme (ID: 461) - 21 points from Silver
- Priya Fischer (ID: 483) - 61 points from Gold
...
**Members 400-500 points away:**
- Omar Dubois (ID: 261) - 401 points from Platinum
- Rosa Okafor (ID: 253) - 405 points from Gold
- Diego Lopez (ID: 387) - 493 points from Gold

All results were fully retrieved (no truncation). The closest member to advancing is
Kira Ekwueme with only 21 points needed to reach Silver tier.
{
  "text": "Here are the members within 500 points of reaching their next loyalty tier (29 total): ...",
  "sql_used": [
    "SELECT member_id, full_name, current_tier, next_tier, points_to_next_tier, current_balance FROM v_balances WHERE points_to_next_tier > 0 AND points_to_next_tier <= 500 ORDER BY points_to_next_tier ASC"
  ],
  "data_caveats": []
}
```

The agent inspected the schema, wrote the SQL itself, and the answer matches the
generator's ground truth: 29 members sit within 500 points of their next tier.

## Every tool call is schema-validated

LLMs emit JSON; databases execute strings. The gap between the two is where
agent bugs live, so strictcall closes it from both directions.

- **Inputs.** Each tool's arguments are a Pydantic model
  ([contracts.py](src/strictcall/contracts.py)). A `limit` outside 1-500 or a
  malformed currency code is rejected before any code runs, and the model gets
  the validation error back to correct its next call.
- **Execution guards.** The SQL tool parses each statement with DuckDB's parser
  and rejects anything that is not exactly one `SELECT`. A wrapping `LIMIT`
  caps rows server-side; a watchdog interrupts runaway queries.
- **Outputs.** Results are validated into `SqlQueryResult` or `FxRateResult`
  before serialization, so the model always sees the same shape: columns, rows,
  a `truncated` flag, backend, timing. Failures come back as a structured
  `ToolError` with a hint, which is what makes the agent's retry loop
  dependable instead of accidental.

## Architecture

```
question ──> agent node (RetryPolicy) ──> tool node ──> agent node ──> answer
                    │                          │
             conversation memory        Pydantic contracts
             (checkpointer, thread_id)  ┌──────┴──────────┐
                                     sql_query         fx_rate
                                     describe_schema  (Frankfurter API)
                                        │
                                  SqlBackend protocol
                                  ┌─────┴───────────┐
                               DuckDB          Snowflake
```

- Hand-built `StateGraph`: a model node with a `RetryPolicy` for transient API
  failures, a conditional edge to the tool node, looping until the model stops
  calling tools.
- Token streaming via `stream_mode="messages"`; conversation memory via a
  checkpointer keyed by `--thread`.
- The tool layer holds one `SqlBackend` reference and never branches on which
  warehouse is behind it. `STRICTCALL_BACKEND=duckdb|snowflake` picks the
  implementation.

## Models

`STRICTCALL_MODEL` selects the model: an Anthropic id (`claude-opus-5` is the
default), or `openrouter:<vendor>/<model>` for anything on OpenRouter. The live
test suite (`tests/test_live.py`, skipped unless `OPENROUTER_API_KEY` is set)
runs the full agent loop against four free-tier models to confirm the tool
contracts hold across providers.

## Dataset

Synthetic loyalty program: `tiers`, `members`, `transactions`, `redemptions`,
and a `v_balances` view with lifetime points, current balance, and points to
the next tier. The generator is deterministic (same seed, same database) and
places a dozen members just under their next tier so the demo questions have
real answers. Nothing is checked in; `python -m strictcall.dataset generate`
rebuilds the warehouse in about a second.

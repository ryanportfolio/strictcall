# strictcall — implementation plan

A LangGraph agent that answers natural-language questions against a SQL warehouse.
Every tool call is validated against an explicit Pydantic contract. One SQL retrieval
interface, two backends: DuckDB (always runnable, zero credentials) and Snowflake
(Phase 3, trial account). Packaged for Amazon Bedrock AgentCore Runtime (Phase 3).

---

## 1. What AgentCore Runtime requires (verified against AWS docs, July 2026)

### Service contract (HTTP protocol)

An agent deployed to AgentCore Runtime must be an HTTP server that:

- Listens on `0.0.0.0:8080`.
- Exposes `POST /invocations` — primary endpoint. JSON request body (AWS examples
  use `{"prompt": "..."}` but the payload shape is application-defined, up to 100MB).
  Response is either plain JSON or `text/event-stream` (SSE) for streaming.
- Exposes `GET /ping` — health check returning `{"status": "Healthy"}` (or
  `HealthyBusy` to keep a session alive during async work). The `bedrock-agentcore`
  SDK handles the ping response automatically.
- Runs on **linux/arm64** (AWS Graviton). Non-negotiable for both packaging modes.

Each user session runs in a dedicated microVM; the caller supplies a
`runtimeSessionId` (minimum 33 characters) on `InvokeAgentRuntime`, and requests
with the same session ID land on the same warm microVM until the idle timeout
(default 15 minutes, configurable via `lifecycleConfiguration`; `maxLifetime` caps
total session age). This maps cleanly onto a LangGraph checkpointer `thread_id`.

Errors from the container surface to callers as HTTP 424 `RuntimeClientError`;
throttling is 429; transient session-provisioning conflicts are 409
`RetryableConflictException` (retryable with backoff).

### Entry point shape

Two supported shapes:

1. **`bedrock-agentcore` Python SDK** (`pip install bedrock-agentcore`):
   ```python
   from bedrock_agentcore import BedrockAgentCoreApp

   app = BedrockAgentCoreApp()


   @app.entrypoint
   async def handler(request):  # sync, async, or (async) generator
       ...  # generator => streamed as SSE automatically


   app.run()  # serves 8080, /invocations + /ping
   ```
2. **Any HTTP framework** (e.g. FastAPI + uvicorn) implementing `/invocations`
   and `/ping` by hand.

We will use shape 1: less code, ping and SSE handled for us, and an async
generator entrypoint plugs directly into LangGraph's `astream`.

### Packaging formats

| Mode | Artifact | Limits | Notes |
|---|---|---|---|
| **Direct code deploy (CodeZip)** | .zip of code + arm64 wheels, uploaded to S3 | 250MB zipped / 750MB unzipped | No Docker needed. `runtime: PYTHON_3_12`…`PYTHON_3_14` supported (Amazon Linux 2023). Dependencies vendored via `uv pip install --python-platform aarch64-manylinux2014 --only-binary=:all: --target=deployment_package`. AWS patches the language runtime. |
| **Container** | arm64 Docker image pushed to ECR | 2GB | Needed only for native deps not available as arm64 wheels. |

Deployment is `bedrock-agentcore-control:CreateAgentRuntime` (boto3) pointing at
the S3 zip (`codeConfiguration`) or ECR image (`containerConfiguration`), plus an
execution role. The AgentCore CLI (`npm install -g @aws/agentcore` —
`agentcore create / dev / deploy / invoke`) automates this and explicitly supports
LangGraph projects; `agentcore dev` runs a local server that mimics the runtime.

**Phase 3 recommendation:** CodeZip. langgraph + langchain + duckdb +
snowflake-connector all ship arm64 manylinux wheels, so no Docker and faster
redeploys. Fall back to container only if the zip exceeds 250MB.

### Pricing

Consumption-based, per-second with a 1-second minimum, spanning microVM boot →
session termination:

- CPU: **$0.0895 per vCPU-hour** — I/O wait and idle time are free if no
  background process runs (i.e. time blocked on LLM responses is not billed).
- Memory: **$0.00945 per GB-hour**, 128MB minimum billing.
- No charge for the AgentCore control plane itself; ECR storage or S3 (CodeZip
  artifact) and data transfer billed separately at standard rates.

A demo-scale portfolio deployment costs effectively pennies; AWS's own example
(10M requests/month, 1 vCPU, 2.5GB peak) is ~$7,235/month, which bounds the
opposite extreme.

### Sources

- Runtime overview: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html
- Service contract: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-service-contract.html
- HTTP protocol contract: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html
- Getting started (CLI): https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-cli.html
- Custom container path: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/getting-started-custom.html
- Direct code deploy: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-code-deploy.html and .../runtime-get-started-code-deploy-python.html
- Supported runtimes: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-code-deploy-supported-runtimes.html
- Pricing: https://aws.amazon.com/bedrock/agentcore/pricing/
- Python SDK: https://github.com/aws/bedrock-agentcore-sdk-python

---

## 2. Tool contract design

Two tools, both defined as LangChain `StructuredTool`s whose argument schemas are
Pydantic models, and whose **return values are also Pydantic models** serialized to
JSON — output validation happens inside the tool before anything reaches the model.

### 2.1 `sql_query` — SQL retrieval

```python
class SqlQueryInput(BaseModel):
    """Contract: what the model must supply."""

    query: str = Field(description="A single read-only SELECT statement.")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum rows to return.")


class SqlQueryResult(BaseModel):
    """Contract: what the tool guarantees back."""

    columns: list[str]
    rows: list[list[str | int | float | bool | None]]
    row_count: int
    truncated: bool  # true if limit clipped the result
    backend: Literal["duckdb", "snowflake"]
    elapsed_ms: float
```

Guards enforced in the tool (not trusted to the model): single statement only,
must parse as a `SELECT` (no DDL/DML/ATTACH/COPY/pragmas), hard row cap applied
via a wrapping `SELECT * FROM (...) LIMIT n`, statement timeout. Validation
failure returns a structured error message to the model so it can self-correct —
this is the retry surface, not an exception.

### 2.2 `describe_schema` — schema introspection

No arguments. Returns:

```python
class TableInfo(BaseModel):
    name: str
    columns: list[ColumnInfo]  # name, type, nullable


class SchemaDescription(BaseModel):
    backend: Literal["duckdb", "snowflake"]
    tables: list[TableInfo]
```

Kept separate from `sql_query` so the agent discovers schema instead of it being
baked into the prompt, and so both backends prove they satisfy the same
introspection contract.

### 2.3 `fx_rate` — external API integration

Currency conversion via the Frankfurter API (https://api.frankfurter.dev — free,
no API key), used for questions like "what is the dollar value of member X's
points in EUR?" (points have a fixed USD redemption value in the dataset).

```python
class FxRateInput(BaseModel):
    base: str = Field(pattern=r"^[A-Z]{3}$")
    target: str = Field(pattern=r"^[A-Z]{3}$")


class FxRateResult(BaseModel):
    base: str
    target: str
    rate: float
    as_of: date
```

HTTP calls go through `httpx` with transport-level retries; non-2xx or schema
drift from the API becomes a structured tool error, never an unhandled exception.

### 2.4 Backend dispatch

```python
class SqlBackend(Protocol):
    name: Literal["duckdb", "snowflake"]

    def execute(self, query: str, limit: int) -> SqlQueryResult: ...
    def describe(self) -> SchemaDescription: ...
```

- `DuckDBBackend` — Phase 2. Opens the generated `.duckdb` file (or builds the
  dataset in-memory for tests).
- `SnowflakeBackend` — Phase 3. Module and class exist from Phase 2 with the
  constructor raising a clear "not implemented until Phase 3" error, so the
  dispatch path is real but the implementation is deferred.

Selection: `STRICTCALL_BACKEND` env var (`duckdb` default). The tool layer holds
one `SqlBackend` reference and never branches on backend type — the Protocol is
the single interface the assignment requires.

### 2.5 Structured final answers

In addition to tool contracts, the agent supports a structured answer mode: the
final response is produced against a JSON schema derived from

```python
class Answer(BaseModel):
    text: str  # the natural-language answer
    sql_used: list[str]  # every query the agent ran
    data_caveats: list[str]  # e.g. "result truncated at 50 rows"
```

via `.with_structured_output(Answer)` on the closing node. The CLI prints
`text`; `--json` dumps the whole object.

---

## 3. Demonstration dataset

Synthetic loyalty program (accepting the default from the brief). Generated by a
committed script — **no data files in git**; anyone rebuilds it with one command.

### Schema

```
tiers        (tier_id PK, name, min_points, points_multiplier)
             -- 4 rows: Bronze 0, Silver 5_000, Gold 15_000, Platinum 40_000
members      (member_id PK, full_name, email, city, joined_at,
              tier_id FK -> tiers)
transactions (txn_id PK, member_id FK, occurred_at, category,
              amount_usd, points_earned)
redemptions  (redemption_id PK, member_id FK, redeemed_at,
              points_spent, reward)
v_balances   VIEW: member_id, points_earned_total, points_spent_total,
              current_balance, current_tier, next_tier, points_to_next_tier
```

### Generation

`python -m strictcall.dataset generate [--seed 42] [--members 500] [--out data/loyalty.duckdb]`

- Seeded `random.Random` → byte-for-byte reproducible; no faker dependency
  (name/city pools inline).
- ~500 members, ~20k transactions over 24 months, redemption behavior skewed by
  tier, a deliberate handful of members within 500 points of the next tier so the
  flagship demo question ("who is within 500 points of the next tier?") has a
  non-trivial answer.
- Writes a DuckDB file (gitignored) or returns an in-memory connection for tests.
- Tier assignment is derived from the generated ledger (not random), so
  cross-table consistency checks in tests are meaningful.

Snowflake (Phase 3) loads the same rows through the same generator via a small
loader, guaranteeing both backends answer identically.

---

## 4. Module layout and dependencies

```
src/strictcall/
    __init__.py
    contracts.py            # all Pydantic I/O models (section 2)
    llm.py                  # model factory: Anthropic API locally,
                            #   Bedrock (langchain-aws) when deployed; fake model in tests
    agent.py                # StateGraph: agent node <-> tool node, RetryPolicy,
                            #   MemorySaver checkpointer, structured-answer node
    tools/
        __init__.py
        sql.py              # sql_query + describe_schema (contract + guards + dispatch)
        fx.py               # fx_rate (httpx, retries)
    backends/
        __init__.py         # get_backend() env-driven factory
        base.py             # SqlBackend Protocol, shared SELECT-only guard
        duckdb_.py          # Phase 2
        snowflake_.py       # Phase 3 stub
    dataset/
        __init__.py
        __main__.py         # CLI: generate
        generate.py
        schema.sql
    cli.py                  # chat REPL: token streaming, --json, --thread for memory
    runtime.py              # Phase 3: BedrockAgentCoreApp entrypoint wrapping agent
tests/
    test_contracts.py       # schema round-trips, validation rejects
    test_backends.py        # DuckDB backend vs contract; guard tests (DML rejected, limit cap)
    test_dataset.py         # determinism (same seed = same data), consistency checks
    test_tools.py           # tool errors are structured, fx via mocked transport
    test_agent.py           # full graph run with scripted fake LLM: tool call ->
                            #   validation error -> self-correct -> final answer; memory across turns
docs/plan.md
.github/workflows/ci.yml    # uv sync, ruff check, ruff format --check, pytest
```

The graph is a hand-built `StateGraph` (not `create_agent`) — the point of the
repo is demonstrating LangGraph construction: agent node with model-call
`RetryPolicy`, conditional edge on tool calls, tool node returning validated
contract JSON, checkpointer for multi-turn memory keyed by `thread_id` (= CLI
`--thread`, = AgentCore `runtimeSessionId` in Phase 3). Token streaming via
`astream(..., stream_mode="messages")`.

### Dependencies (Phase 2)

Runtime: `langgraph` (1.x), `langchain-core`, `langchain-anthropic`,
`pydantic` (2.x), `duckdb`, `httpx`.
Dev: `pytest`, `pytest-asyncio`, `ruff`.
Phase 3 adds: `bedrock-agentcore`, `langchain-aws`, `snowflake-connector-python`.

Python `>=3.12` (matches AgentCore `PYTHON_3_12`+ direct-deploy runtimes), `uv`
managed, `uv.lock` committed.

Retries, concretely: (a) model node — LangGraph `RetryPolicy` on transient API
errors; (b) tool level — Pydantic validation errors returned to the model as tool
messages so the agent retries with corrected arguments (asserted in
`test_agent.py`); (c) HTTP — `httpx` transport retries for the fx tool.

---

## 5. Open questions / unverified items

1. **Model access for local runs.** The live demo run needs a real model.
   Assumed: `ANTHROPIC_API_KEY` available locally (`claude-sonnet-5` default,
   overridable via `STRICTCALL_MODEL`). Confirm before Phase 2's "real question
   answered" acceptance step; tests themselves need no key (fake model).
2. **`bedrock-agentcore` SDK minimum Python** — the repo README doesn't state a
   floor explicitly; runtime-side support is Python 3.12–3.14, so `>=3.12` is
   safe, but verify the pin when Phase 3 adds the dependency.
3. **CodeZip size** — langgraph + langchain + duckdb arm64 wheels almost
   certainly fit 250MB zipped, but not measured. If snowflake-connector pushes
   past it, packaging falls back to the container path (section 1).
4. **Snowflake trial** — account URL/credentials and whether key-pair auth is
   available: unknown until Phase 3.
5. **Payload shape at the runtime boundary** — `/invocations` body is
   application-defined; we'll mirror AWS's `{"prompt": ...}` convention plus our
   session mapping. Exact response envelope for streaming chunks is our choice
   (SSE `data:` lines of JSON); nothing in the contract constrains it further.
6. **AgentCore region availability and free-tier credits** ($200 new-account
   credit noted on the pricing page) — not load-bearing for the design; verify
   region (default us-west-2 in AWS tooling) when deploying.

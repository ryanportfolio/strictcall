"""Hand-built HTTP runtime implementing the Amazon Bedrock AgentCore Runtime contract.

AgentCore Runtime talks to an agent over plain HTTP: POST /invocations with a
JSON payload, GET /ping for health, and a session-id header that scopes
conversation state. This module implements that contract directly with FastAPI
instead of the bedrock-agentcore SDK, so the same container runs locally, on
any container host, or on AgentCore itself.

Run locally:

    uvicorn strictcall.runtime:app --host 0.0.0.0 --port 8080
"""

import threading
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from strictcall.agent import build_agent, collect_answer
from strictcall.backends import get_backend
from strictcall.contracts import Answer
from strictcall.llm import get_chat_model

# AgentCore forwards its runtimeSessionId (>= 33 chars) in this header; each
# distinct value gets its own conversation memory via the LangGraph thread_id.
SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
DEFAULT_SESSION = "default"


class InvocationRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Natural-language question for the agent.")


class BusyCounter:
    """Tracks in-flight invocations for /ping's Healthy vs HealthyBusy."""

    def __init__(self) -> None:
        self._count = 0
        self._lock = threading.Lock()

    def __enter__(self) -> "BusyCounter":
        with self._lock:
            self._count += 1
        return self

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self._count -= 1

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._count > 0


def create_app(backend=None, model=None) -> FastAPI:
    """Build the runtime app. Pass a backend/model to skip env-based defaults
    (tests inject a fake model this way)."""
    busy = BusyCounter()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.agent = build_agent(backend or get_backend(), model=model or get_chat_model())
        yield

    app = FastAPI(title="strictcall runtime", lifespan=lifespan)

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"status": "HealthyBusy" if busy.busy else "Healthy"}

    @app.post("/invocations")
    def invocations(
        request: InvocationRequest,
        session_id: Annotated[str | None, Header(alias=SESSION_HEADER)] = None,
    ) -> Answer:
        thread_id = session_id or DEFAULT_SESSION
        with busy:
            result = app.state.agent.invoke(
                {"messages": [HumanMessage(request.prompt)]},
                {"configurable": {"thread_id": thread_id}},
            )
        return collect_answer(result["messages"])

    return app


app = create_app()

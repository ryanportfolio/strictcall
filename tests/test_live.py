"""Live integration tests: the real agent loop against real hosted models.

Opt-in: they run only when OPENROUTER_API_KEY is set, so CI and offline runs
skip them automatically. Most listed models are on OpenRouter's free tier;
any slug without a `:free` suffix is billed per call.
"""

import os

import pytest
from langchain_core.messages import HumanMessage

from strictcall.agent import build_agent, collect_answer
from strictcall.backends.duckdb_ import DuckDBBackend
from strictcall.llm import get_chat_model

LIVE_MODELS = [
    "openrouter:cohere/north-mini-code:free",
    "openrouter:inclusionai/ling-3.0-flash",  # paid: the :free variant was withdrawn
    "openrouter:poolside/laguna-xs-2.1:free",
    "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free",
]

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="live tests need OPENROUTER_API_KEY",
)


@pytest.mark.parametrize("model", LIVE_MODELS)
def test_agent_answers_a_count_question_with_sql(demo_con, model):
    agent = build_agent(DuckDBBackend(connection=demo_con), model=get_chat_model(model))
    result = agent.invoke(
        {"messages": [HumanMessage("Exactly how many loyalty members are in the warehouse?")]},
        {"configurable": {"thread_id": f"live-{model}"}},
    )
    answer = collect_answer(result["messages"])
    assert answer.sql_used, "expected the model to reach for the sql_query tool"
    assert "60" in answer.text  # the test warehouse has exactly 60 members

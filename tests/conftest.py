from collections.abc import Iterator

import duckdb
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from strictcall.backends.duckdb_ import DuckDBBackend
from strictcall.dataset import generate

TEST_SEED = 7
TEST_MEMBERS = 60


class ScriptedChatModel(GenericFakeChatModel):
    """Fake chat model that replays a fixed list of AI messages, including tool
    calls. Lets the full graph run with no API key."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(messages: list[AIMessage]) -> ScriptedChatModel:
    script: Iterator[AIMessage] = iter(messages)
    return ScriptedChatModel(messages=script)


@pytest.fixture(scope="session")
def demo_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    generate(con, seed=TEST_SEED, members=TEST_MEMBERS)
    return con


@pytest.fixture
def backend(demo_con) -> DuckDBBackend:
    return DuckDBBackend(connection=demo_con)

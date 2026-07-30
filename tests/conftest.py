import json
from collections.abc import Iterator

import duckdb
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from strictcall.backends.duckdb_ import DuckDBBackend
from strictcall.dataset import generate

TEST_SEED = 7
TEST_MEMBERS = 60


class ScriptedChatModel(GenericFakeChatModel):
    """Fake chat model that replays a fixed list of AI messages, including tool
    calls. Lets the full graph run with no API key."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # The parent's _stream yields nothing for empty-content tool-call
        # messages, which breaks streaming runs. Replay each scripted message
        # as one chunk instead.
        message = next(self.messages)
        if isinstance(message, AIMessage) and message.tool_calls:
            chunk = AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": call["name"],
                        "args": json.dumps(call["args"]),
                        "id": call["id"],
                        "index": i,
                        "type": "tool_call_chunk",
                    }
                    for i, call in enumerate(message.tool_calls)
                ],
            )
        else:
            chunk = AIMessageChunk(content=message.content)
        yield ChatGenerationChunk(message=chunk)


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

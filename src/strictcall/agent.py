"""The LangGraph agent: a hand-built StateGraph with tool calling, retries,
and conversation memory.

Graph shape:

    START -> agent -> (tool calls?) -> tools -> agent -> ... -> END

- The agent node carries a RetryPolicy for transient model/API failures.
- Tool-argument mistakes are handled one level down: tools return a structured
  ToolError message and the model corrects itself on the next loop.
- An InMemorySaver checkpointer keyed by thread_id provides multi-turn memory.
"""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from strictcall.backends.base import SqlBackend
from strictcall.contracts import Answer
from strictcall.llm import get_chat_model
from strictcall.tools import make_fx_tool, make_sql_tools

SYSTEM_PROMPT = (
    "You are strictcall, an analyst for a loyalty-program SQL warehouse.\n"
    "- Call describe_schema before querying tables you have not seen this conversation.\n"
    "- Answer with numbers taken from query results, never from guesses.\n"
    "- Points redeem at a fixed value of 1 cent (USD) per point; use fx_rate for "
    "other currencies.\n"
    "- If a tool returns an error, fix your call and try again.\n"
    "- Mention it when a result was truncated at the row limit."
)


def build_agent(
    backend: SqlBackend,
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    llm = model or get_chat_model()
    tools = [*make_sql_tools(backend), make_fx_tool()]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node, retry_policy=RetryPolicy(max_attempts=3))
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def message_text(message: BaseMessage) -> str:
    """Flatten string-or-blocks message content to plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def collect_answer(messages: Sequence[BaseMessage]) -> Answer:
    """Assemble the structured Answer for the latest turn, without an extra
    model call: the text is the final AI message, sql_used and data_caveats are
    read back from the validated tool traffic."""
    last_human = max(
        (i for i, m in enumerate(messages) if m.type == "human"),
        default=0,
    )
    turn = messages[last_human:]
    sql_used: list[str] = []
    caveats: list[str] = []
    for message in turn:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["name"] == "sql_query":
                    sql_used.append(call["args"].get("query", ""))
        if message.type == "tool" and '"truncated":true' in message_text(message).replace(" ", ""):
            caveats.append("At least one query result was truncated at the row limit.")
            break
    final = next((m for m in reversed(turn) if isinstance(m, AIMessage) and not m.tool_calls), None)
    return Answer(
        text=message_text(final) if final else "",
        sql_used=sql_used,
        data_caveats=caveats,
    )

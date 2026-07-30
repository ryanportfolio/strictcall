from conftest import ScriptedChatModel, scripted
from langchain_core.messages import AIMessage, HumanMessage

from strictcall.agent import build_agent, collect_answer, message_text


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def test_agent_self_corrects_after_structured_tool_error(backend):
    script = [
        tool_call("sql_query", {"query": "DROP TABLE members", "limit": 5}, "c1"),
        tool_call("sql_query", {"query": "SELECT COUNT(*) AS n FROM members", "limit": 5}, "c2"),
        AIMessage("There are 60 members."),
    ]
    agent = build_agent(backend, model=scripted(script))
    result = agent.invoke(
        {"messages": [HumanMessage("How many members are there?")]},
        {"configurable": {"thread_id": "t1"}},
    )
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 2
    assert '"error"' in message_text(tool_messages[0])  # rejected DDL, agent retried
    assert '"columns"' in message_text(tool_messages[1])  # corrected call succeeded

    answer = collect_answer(result["messages"])
    assert answer.text == "There are 60 members."
    assert answer.sql_used == [
        "DROP TABLE members",
        "SELECT COUNT(*) AS n FROM members",
    ]


def test_agent_uses_schema_tool_output(backend):
    script = [
        tool_call("describe_schema", {}, "c1"),
        AIMessage("The warehouse has tiers, members, transactions, redemptions, v_balances."),
    ]
    agent = build_agent(backend, model=scripted(script))
    result = agent.invoke(
        {"messages": [HumanMessage("What tables exist?")]},
        {"configurable": {"thread_id": "t2"}},
    )
    tool_message = next(m for m in result["messages"] if m.type == "tool")
    assert '"v_balances"' in message_text(tool_message)


def test_memory_persists_across_turns_on_same_thread(backend):
    agent = build_agent(
        backend, model=scripted([AIMessage("First answer."), AIMessage("Second answer.")])
    )
    config = {"configurable": {"thread_id": "memory"}}
    agent.invoke({"messages": [HumanMessage("first question")]}, config)
    result = agent.invoke({"messages": [HumanMessage("second question")]}, config)
    types = [m.type for m in result["messages"]]
    assert types == ["human", "ai", "human", "ai"]
    assert message_text(result["messages"][0]) == "first question"


def test_threads_are_isolated(backend):
    agent = build_agent(backend, model=scripted([AIMessage("A."), AIMessage("B.")]))
    agent.invoke({"messages": [HumanMessage("thread one")]}, {"configurable": {"thread_id": "one"}})
    result = agent.invoke(
        {"messages": [HumanMessage("thread two")]}, {"configurable": {"thread_id": "two"}}
    )
    assert len(result["messages"]) == 2


class TransientFailure(Exception):
    pass


class FlakyModel(ScriptedChatModel):
    failures_left: int = 1

    def _generate(self, *args, **kwargs):
        if self.failures_left > 0:
            self.failures_left -= 1
            raise TransientFailure("simulated transient API failure")
        return super()._generate(*args, **kwargs)


def test_model_node_retries_transient_failures(backend):
    model = FlakyModel(messages=iter([AIMessage("Recovered.")]), failures_left=1)
    agent = build_agent(backend, model=model)
    result = agent.invoke(
        {"messages": [HumanMessage("hello")]}, {"configurable": {"thread_id": "retry"}}
    )
    assert message_text(result["messages"][-1]) == "Recovered."

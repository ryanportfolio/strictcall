"""Contract tests for the AgentCore-compatible HTTP runtime.

These boot the real FastAPI app with a scripted model, so the full
request -> agent -> tool -> response path runs offline with no API key.
"""

from conftest import scripted
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from strictcall.runtime import DEFAULT_SESSION, SESSION_HEADER, BusyCounter, create_app


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def client_for(backend, script: list[AIMessage]) -> TestClient:
    return TestClient(create_app(backend=backend, model=scripted(script)))


def test_ping_reports_healthy(backend):
    with client_for(backend, []) as client:
        response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "Healthy"}


def test_invocation_runs_the_agent_and_returns_a_validated_answer(backend):
    script = [
        tool_call("sql_query", {"query": "SELECT COUNT(*) AS n FROM members", "limit": 5}, "c1"),
        AIMessage("There are 60 members."),
    ]
    with client_for(backend, script) as client:
        response = client.post("/invocations", json={"prompt": "How many members are there?"})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "There are 60 members."
    assert body["sql_used"] == ["SELECT COUNT(*) AS n FROM members"]
    assert body["data_caveats"] == []


def test_sessions_map_to_isolated_conversation_threads(backend):
    script = [AIMessage("Answer for A."), AIMessage("Answer for B.")]
    with client_for(backend, script) as client:
        client.post(
            "/invocations", json={"prompt": "question A"}, headers={SESSION_HEADER: "session-a"}
        )
        client.post(
            "/invocations", json={"prompt": "question B"}, headers={SESSION_HEADER: "session-b"}
        )
        agent = client.app.state.agent
        state_a = agent.get_state({"configurable": {"thread_id": "session-a"}})
        state_b = agent.get_state({"configurable": {"thread_id": "session-b"}})
        state_default = agent.get_state({"configurable": {"thread_id": DEFAULT_SESSION}})
    assert len(state_a.values["messages"]) == 2  # human + ai, no bleed from B
    assert len(state_b.values["messages"]) == 2
    assert state_default.values == {}  # header absent was never used


def test_missing_session_header_falls_back_to_default_thread(backend):
    with client_for(backend, [AIMessage("Hi.")]) as client:
        client.post("/invocations", json={"prompt": "hello"})
        state = client.app.state.agent.get_state({"configurable": {"thread_id": DEFAULT_SESSION}})
    assert len(state.values["messages"]) == 2


def test_invalid_payload_is_rejected_before_the_agent_runs(backend):
    with client_for(backend, []) as client:
        empty = client.post("/invocations", json={"prompt": ""})
        missing = client.post("/invocations", json={})
    assert empty.status_code == 422
    assert missing.status_code == 422


def test_streaming_invocation_emits_deltas_then_one_final_answer(backend):
    import json

    script = [
        tool_call("sql_query", {"query": "SELECT COUNT(*) AS n FROM members", "limit": 5}, "c1"),
        AIMessage("There are 60 members."),
    ]
    with client_for(backend, script) as client:
        with client.stream(
            "POST", "/invocations", json={"prompt": "How many members?", "stream": True}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join(response.iter_text())
    events = [json.loads(line[len("data: ") :]) for line in body.split("\n\n") if line]
    deltas = "".join(e["delta"] for e in events if "delta" in e)
    finals = [e["answer"] for e in events if "answer" in e]
    assert "There are 60 members." in deltas
    assert len(finals) == 1
    assert finals[0]["sql_used"] == ["SELECT COUNT(*) AS n FROM members"]
    assert finals[0]["text"] == "There are 60 members."


def test_streaming_reports_a_rejected_tool_call_before_the_retry(backend):
    import json

    script = [
        tool_call("sql_query", {"query": "DROP TABLE members", "limit": 5}, "c1"),
        tool_call("sql_query", {"query": "SELECT COUNT(*) AS n FROM members", "limit": 5}, "c2"),
        AIMessage("There are 60 members."),
    ]
    with client_for(backend, script) as client:
        with client.stream(
            "POST", "/invocations", json={"prompt": "How many members?", "stream": True}
        ) as response:
            body = "".join(response.iter_text())
    events = [json.loads(line[len("data: ") :]) for line in body.split("\n\n") if line]
    rejected = [e["tool_error"] for e in events if "tool_error" in e]
    assert len(rejected) == 1
    assert rejected[0]["tool"] == "sql_query"
    assert "Only SELECT statements are allowed" in rejected[0]["error"]

    final = next(e["answer"] for e in events if "answer" in e)
    assert [f["tool"] for f in final["tool_errors"]] == ["sql_query"]


def test_root_serves_the_chat_ui(backend):
    with client_for(backend, []) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "strictcall" in response.text
    assert "/invocations" in response.text  # the UI talks to the real contract


def test_busy_counter_tracks_in_flight_work():
    counter = BusyCounter()
    assert not counter.busy
    with counter:
        assert counter.busy
        with counter:
            assert counter.busy
        assert counter.busy
    assert not counter.busy

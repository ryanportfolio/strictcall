"""Chat CLI with token streaming.

strictcall "Which members are within 500 points of the next tier?"
strictcall --json "How many Platinum members are there?"
strictcall            # interactive REPL, memory persists across turns
"""

import argparse
import sys

from langchain_core.messages import AIMessageChunk, HumanMessage

from strictcall.agent import build_agent, collect_answer, message_text, tool_failure
from strictcall.backends import get_backend
from strictcall.llm import get_chat_model


def run_turn(agent, question: str, thread_id: str, as_json: bool) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    for chunk, metadata in agent.stream(
        {"messages": [HumanMessage(question)]}, config, stream_mode="messages"
    ):
        if metadata.get("langgraph_node") == "tools":
            failure = tool_failure(chunk)
            if failure:
                print(f"[{failure.tool} rejected: {failure.error}]", file=sys.stderr, flush=True)
            continue
        if isinstance(chunk, AIMessageChunk):
            for call in chunk.tool_calls:
                if call.get("name"):
                    print(f"\n[{call['name']}]", file=sys.stderr, flush=True)
            text = message_text(chunk)
            if text:
                print(text, end="", flush=True)
    print()
    if as_json:
        state = agent.get_state(config)
        print(collect_answer(state.values["messages"]).model_dump_json(indent=2))


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="strictcall")
    parser.add_argument("question", nargs="*", help="Question to ask; omit for a REPL.")
    parser.add_argument("--json", action="store_true", help="Also print the structured Answer.")
    parser.add_argument("--thread", default="cli", help="Conversation thread id (memory key).")
    parser.add_argument("--db", default=None, help="Path to the DuckDB file.")
    parser.add_argument("--model", default=None, help="Model id override.")
    args = parser.parse_args(argv)

    backend = get_backend(args.db)
    agent = build_agent(backend, model=get_chat_model(args.model))

    if args.question:
        run_turn(agent, " ".join(args.question), args.thread, args.json)
        return

    print("strictcall REPL - Ctrl+C or empty line to exit.")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        run_turn(agent, question, args.thread, args.json)


if __name__ == "__main__":
    main()

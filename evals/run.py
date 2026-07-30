"""Golden-question evaluation harness.

Scores the full agent loop against ground truth computed live from the
warehouse, so expectations never go stale when the generator changes. Each
question has a reference SQL query; the agent passes when it actually used
sql_query and its answer contains the reference value.

    uv run python evals/run.py --models "openrouter:cohere/north-mini-code:free"

Writes a markdown report to evals/RESULTS.md (override with --out).
"""

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
from langchain_core.messages import HumanMessage

from strictcall.agent import build_agent, collect_answer
from strictcall.backends.duckdb_ import DuckDBBackend
from strictcall.llm import get_chat_model


@dataclass(frozen=True)
class Golden:
    question: str
    truth_sql: str  # must return a single integer scalar


GOLDEN = [
    Golden(
        "Exactly how many loyalty members are in the warehouse?",
        "SELECT COUNT(*) FROM members",
    ),
    Golden(
        "How many Platinum members are there?",
        "SELECT COUNT(*) FROM v_balances WHERE current_tier = 'Platinum'",
    ),
    Golden(
        "How many members are within 500 points of reaching the next tier?",
        "SELECT COUNT(*) FROM v_balances "
        "WHERE points_to_next_tier > 0 AND points_to_next_tier <= 500",
    ),
    Golden(
        "What is the highest current point balance any single member holds?",
        "SELECT MAX(current_balance) FROM v_balances",
    ),
    Golden(
        "How many redemptions have been made in total across all members?",
        "SELECT COUNT(*) FROM redemptions",
    ),
    Golden(
        "How many tiers does the loyalty program have?",
        "SELECT COUNT(*) FROM tiers",
    ),
    Golden(
        "How many lifetime points does a member need to reach the Gold tier?",
        "SELECT min_points FROM tiers WHERE name = 'Gold'",
    ),
    Golden(
        "How many members joined in 2025?",
        "SELECT COUNT(*) FROM members WHERE YEAR(joined_at) = 2025",
    ),
]


def answer_contains(text: str, expected: int) -> bool:
    cleaned = text.replace(",", "")  # tolerate 15,000 vs 15000
    return re.search(rf"(?<!\d){expected}(?!\d)", cleaned) is not None


def run_model(model_name: str, db_path: str) -> list[dict]:
    con = duckdb.connect(db_path, read_only=True)
    backend = DuckDBBackend(connection=con)
    rows = []
    for i, golden in enumerate(GOLDEN):
        expected = int(con.sql(golden.truth_sql).fetchone()[0])
        agent = build_agent(backend, model=get_chat_model(model_name))
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(golden.question)]},
                {"configurable": {"thread_id": f"eval-{i}"}},
            )
            answer = collect_answer(result["messages"])
            used_sql = bool(answer.sql_used)
            correct = used_sql and answer_contains(answer.text, expected)
            note = "" if correct else ("no sql_query call" if not used_sql else "wrong value")
        except Exception as exc:  # a hard failure is a scored miss, not a crash
            correct, note = False, f"error: {type(exc).__name__}"
        rows.append(
            {"question": golden.question, "expected": expected, "pass": correct, "note": note}
        )
        status = "PASS" if correct else f"FAIL ({note})"
        print(f"  [{i + 1}/{len(GOLDEN)}] {status}: {golden.question}", flush=True)
    return rows


def report(results: dict[str, list[dict]], db_path: str) -> str:
    lines = [
        "# Eval results",
        "",
        f"Run on {date.today().isoformat()} against the deterministic demo warehouse "
        f"(`{db_path}`, seed 42, 500 members). Ground truth is computed from the "
        "warehouse at run time by `evals/run.py`; a question passes only when the "
        "agent both called `sql_query` and answered with the reference value.",
        "",
    ]
    for model, rows in results.items():
        passed = sum(r["pass"] for r in rows)
        lines += [f"## `{model}` - {passed}/{len(rows)}", ""]
        lines += ["| Question | Expected | Result |", "|---|---|---|"]
        for r in rows:
            outcome = "pass" if r["pass"] else f"fail ({r['note']})"
            lines.append(f"| {r['question']} | {r['expected']} | {outcome} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="Model ids to evaluate.")
    parser.add_argument("--db", default="data/loyalty.duckdb")
    parser.add_argument("--out", default="evals/RESULTS.md")
    args = parser.parse_args()

    results = {}
    for model in args.models:
        print(f"Evaluating {model}", flush=True)
        results[model] = run_model(model, args.db)

    markdown = report(results, args.db)
    Path(args.out).write_text(markdown, encoding="utf-8")
    print(f"\nReport written to {args.out}")
    for model, rows in results.items():
        print(f"{model}: {sum(r['pass'] for r in rows)}/{len(rows)}")
    if any(not r["pass"] for rows in results.values() for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()

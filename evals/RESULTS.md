# Eval results

Run on 2026-07-30 against the deterministic demo warehouse (`data/loyalty.duckdb`, seed 42, 500 members). Ground truth is computed from the warehouse at run time by `evals/run.py`; a question passes only when the agent both called `sql_query` and answered with the reference value.

## `openrouter:cohere/north-mini-code:free` - 8/8

| Question | Expected | Result |
|---|---|---|
| Exactly how many loyalty members are in the warehouse? | 500 | pass |
| How many Platinum members are there? | 1 | pass |
| How many members are within 500 points of reaching the next tier? | 29 | pass |
| What is the highest current point balance any single member holds? | 39831 | pass |
| How many redemptions have been made in total across all members? | 742 | pass |
| How many tiers does the loyalty program have? | 4 | pass |
| How many lifetime points does a member need to reach the Gold tier? | 15000 | pass |
| How many members joined in 2025? | 251 | pass |

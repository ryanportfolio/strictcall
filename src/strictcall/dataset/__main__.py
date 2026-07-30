"""CLI: python -m strictcall.dataset generate [--seed N] [--members N] [--out PATH]"""

import argparse

from strictcall.dataset.generate import build_database


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m strictcall.dataset")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Build the demo loyalty database.")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--members", type=int, default=500)
    gen.add_argument("--out", default="data/loyalty.duckdb")
    args = parser.parse_args()

    path = build_database(args.out, seed=args.seed, members=args.members)
    print(f"Wrote {path} (seed={args.seed}, members={args.members})")


if __name__ == "__main__":
    main()

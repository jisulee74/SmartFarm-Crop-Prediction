from __future__ import annotations

import argparse
import json

from .config import DEFAULT_OUTPUT
from .finalize import build_rankings, consolidate_test_outputs
from .prepare import prepare
from .runner import initialize, run, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "initialize", "run", "all", "status", "rank", "consolidate"),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "initialize":
        result = initialize(args.output)
    elif args.command in {"run", "all"}:
        result = run(args.output)
    elif args.command == "rank":
        result = build_rankings(args.output)
    elif args.command == "consolidate":
        result = consolidate_test_outputs(args.output)
    else:
        result = status(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

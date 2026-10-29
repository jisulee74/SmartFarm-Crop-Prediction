from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_OUTPUT
from .report import report
from .runner import initialize, run, status


def main():
    parser = argparse.ArgumentParser(description="Semantic E6 v2 follow-up campaign")
    parser.add_argument("command", choices=["all", "prepare", "run", "report", "status", "stop"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.command == "prepare":
        initialize(root)
    elif args.command in {"all", "run"}:
        state = run(root)
        if state == "finished":
            report(root)
    elif args.command == "report":
        report(root)
    elif args.command == "stop":
        root.mkdir(parents=True, exist_ok=True)
        (root / "STOP").touch()
        print("Stop requested; workers stop at the next persisted combination boundary")
    print(json.dumps(status(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

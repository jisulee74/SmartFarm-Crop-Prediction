from __future__ import annotations

import argparse
import json

from .config import DEFAULT_OUTPUT
from .runner import initialize, run, status


def main():
    parser = argparse.ArgumentParser(description="Audit E1/E2 and backfill the 96 missing combinations")
    parser.add_argument("command", choices=["all", "audit", "prepare", "run", "status", "stop"])
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    from pathlib import Path
    root = Path(args.output).resolve()
    if args.command in {"audit", "prepare"}:
        initialize(root)
    elif args.command in {"all", "run"}:
        run(root)
    elif args.command == "stop":
        root.mkdir(parents=True, exist_ok=True)
        (root / "STOP").touch()
        print("Stop requested; workers stop at the next persisted combination boundary")
    print(json.dumps(status(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

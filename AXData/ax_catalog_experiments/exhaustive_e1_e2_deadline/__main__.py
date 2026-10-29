from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_OUTPUT
from .runner import finalize, run, run_model_phase, status


def main():
    parser = argparse.ArgumentParser(description="Deadline screening for all E1-E7 combinations")
    parser.add_argument("command", choices=("run", "gpu", "cpu", "status", "finalize"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if args.command == "run":
        run(root)
    elif args.command in {"gpu", "cpu"}:
        run_model_phase(root, args.command)
    elif args.command == "finalize":
        finalize(root)
    print(json.dumps(status(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
import json
from pathlib import Path
from .config import DEFAULT_OUTPUT
from .report import report
from .runner import initialize, run, status


def main():
    p = argparse.ArgumentParser(description="Exhaustive C0 + non-empty E1-E7 campaign")
    p.add_argument(
        "command", choices=["all", "prepare", "run", "report", "status", "stop"]
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args()
    root = a.output.resolve()
    if a.command == "prepare":
        initialize(root)
    elif a.command in ("all", "run"):
        state = run(root)
        if state == "finished":
            report(root)
    elif a.command == "report":
        report(root)
    elif a.command == "stop":
        root.mkdir(parents=True, exist_ok=True)
        (root / "STOP").touch()
        print("Stop requested; workers stop at the next persisted combination boundary")
    print(json.dumps(status(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

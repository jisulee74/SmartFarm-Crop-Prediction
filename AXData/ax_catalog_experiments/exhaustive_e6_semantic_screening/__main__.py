from __future__ import annotations

import argparse
import json
from pathlib import Path

from .finalize import finalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("finalize",))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = finalize(Path(args.output).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

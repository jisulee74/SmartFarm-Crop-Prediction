from __future__ import annotations

import os
from pathlib import Path

from ..catalog import AX


LEGACY_ENV = AX.parent / "GEAS" / "GEAS3.5" / ".env"
DB_KEYS = ("HOST", "PORT", "USER", "PASSWORD")


def load_db_environment(path: Path = LEGACY_ENV) -> Path | None:
    """Load legacy GEAS DB settings without overriding explicit run settings."""
    if os.getenv("FARMSTOM_DB_PASSWORD"):
        return None
    path = Path(path)
    if not path.is_file():
        return None
    allowed = {f"GEAS_DB_{suffix}" for suffix in DB_KEYS}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in allowed:
            values[key] = value.strip().strip("\"'")
    for suffix in DB_KEYS:
        value = values.get(f"GEAS_DB_{suffix}")
        if value:
            os.environ.setdefault(f"FARMSTOM_DB_{suffix}", value)
    os.environ.setdefault("FARMSTOM_DB_NAME", "farmstom_usoo")
    return path if os.getenv("FARMSTOM_DB_PASSWORD") else None

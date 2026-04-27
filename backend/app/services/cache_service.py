from datetime import UTC, datetime, timedelta
from typing import Any


class TTLCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[datetime, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None

        expires_at, value = entry
        if datetime.now(UTC) >= expires_at:
            self._store.pop(key, None)
            return None

        return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
        self._store[key] = (expires_at, value)


cache = TTLCache()

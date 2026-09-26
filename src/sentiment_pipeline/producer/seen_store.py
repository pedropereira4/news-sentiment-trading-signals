"""Bounded, file-backed record of already-published article ids (shared by all producers)."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path

log = logging.getLogger(__name__)


class SeenStore:
    """Bounded, file-backed LRU of article ids so restarts don't re-publish everything."""

    def __init__(self, path: str, max_size: int) -> None:
        self.path = Path(path)
        self.max_size = max_size
        self._ids: OrderedDict[str, None] = OrderedDict()
        if self.path.exists():
            try:
                self._ids = OrderedDict.fromkeys(json.loads(self.path.read_text()))
            except (json.JSONDecodeError, OSError):
                log.warning("Could not read %s, starting with empty seen-store", self.path)

    def __contains__(self, key: str) -> bool:
        return key in self._ids

    def add(self, key: str) -> None:
        self._ids[key] = None
        self._ids.move_to_end(key)
        while len(self._ids) > self.max_size:
            self._ids.popitem(last=False)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(list(self._ids)))
        tmp.replace(self.path)  # atomic

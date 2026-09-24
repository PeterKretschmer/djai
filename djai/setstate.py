"""Crash-safe set state: the cue queue, what has played, and the plan.

THREADING CONTEXT: control thread only. Every write is a whole-file replace:
the new state is written to a temporary file beside the old one, flushed to
disk, then swapped in with ``os.replace``, which is atomic on the same volume.
A process killed at any instant leaves either the old file or the new one,
never half of either.

Only a session given a path persists anything; ``cmd_play`` gives it one.
A session that ended cleanly marks the file ``ended``, and the next start
begins a fresh set; a killed one does not, and the next start resumes it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Bumped when a field changes meaning. Readers ignore fields they do not know
#: and default the ones that are missing, so an older file still loads.
STATE_VERSION: int = 1


@dataclass
class SetState:
    path: Path | None = None
    version: int = STATE_VERSION
    started_at: float = field(default_factory=time.time)
    ended: bool = False
    #: Operator cues, in queue order. Each is a dict: id, track_id, title,
    #: mode ("next" | "after" | "now"), after (tracks still to wait),
    #: status ("queued" | "needs_bridge"), bridge (None | "tempo" | "track"),
    #: warning (text or None), added_at.
    cue_queue: list[dict] = field(default_factory=list)
    next_cue_id: int = 1
    #: Track ids, oldest first.
    history: list[str] = field(default_factory=list)
    persona: str = "default"
    set_seed: int = 0
    #: The last set plan, as `planner.Plan.as_dict()`.
    plan: dict | None = None

    def save(self) -> None:
        """Atomically replace the file. No-op without a path."""
        if self.path is None:
            return
        data = asdict(self)
        data.pop("path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    @classmethod
    def load(cls, path: Path) -> "SetState":
        """Read a saved state. A missing or unreadable file gives a fresh one."""
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls(path=path)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        known.pop("path", None)
        state = cls(path=path, **known)
        return state

    def add_cue(self, entry: dict, front: bool = False) -> dict:
        entry = dict(entry, id=self.next_cue_id)
        self.next_cue_id += 1
        if front:
            self.cue_queue.insert(0, entry)
        else:
            self.cue_queue.append(entry)
        self.save()
        return entry

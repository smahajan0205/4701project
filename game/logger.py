"""JSON-lines session logs: one file per run, one JSON object per line.

Record types (see docs/game.md for every field):
  session_start  config, input mode, participant, condition, git commit
  input          every gesture/keyboard event received, accepted or not, with
                 its full timestamp chain (capture -> emit -> recv -> applied -> displayed)
  frame          per-frame player state and frame timing
  spawn / crash / pass / end   world events from World.step()
  session_end    summary, plus frame-timing statistics
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=Path(__file__).parent, timeout=2)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def session_path(participant: str, input_mode: str, seed: int, log_dir: Path = LOG_DIR) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return log_dir / f"{stamp}_{participant}_{input_mode}_seed{seed}.jsonl"


class SessionLogger:
    def __init__(self, path: Path, header: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = open(path, "w")
        self.write({"type": "session_start",
                    "wall_time": dt.datetime.now().isoformat(timespec="seconds"),
                    "git": git_commit(), **header})

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self, summary: dict | None = None) -> None:
        if self._f.closed:
            return
        if summary is not None:
            self.write({"type": "session_end", **summary})
        self._f.close()


def read_log(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

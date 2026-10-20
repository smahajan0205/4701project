"""Shared event interface between the perception/model side and the game.

This is the contract agreed in week 1: the classifier emits
``(event_type, confidence, timestamp)`` and the game consumes it. Neither side
imports the other's code; they only share this module.

Transport is JSON over UDP on localhost, so the classifier can run in its own
process (camera + MediaPipe + model) while the game runs in another. Minimal
sender usage::

    from common.events import EventSender, EventType, now

    sender = EventSender()                    # 127.0.0.1:5005 by default
    t_cap = now()                             # right after cap.read()
    ...                                       # landmarks, features, model
    sender.send(EventType.LEFT, confidence=0.93, t_capture=t_cap)

All timestamps come from ``now()`` (``time.monotonic``). On macOS and Linux
that clock is shared across processes on the same machine, so the game can
subtract the classifier's timestamps from its own to decompose latency.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict, dataclass, field
from enum import Enum

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5005


def now() -> float:
    """The single clock used for every timestamp in the system (seconds)."""
    return time.monotonic()


class EventType(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    JUMP = "jump"
    ROLL = "roll"

    @classmethod
    def parse(cls, name: str) -> "EventType":
        """Accept the canonical names plus gesture-class aliases (up/down/...)."""
        key = name.strip().lower()
        key = _ALIASES.get(key, key)
        return cls(key)


_ALIASES = {
    "up": "jump",
    "down": "roll",
    "swipe_left": "left",
    "swipe_right": "right",
    "swipe_up": "jump",
    "swipe_down": "roll",
    "l": "left",
    "r": "right",
    "u": "jump",
    "d": "roll",
}


@dataclass(frozen=True)
class GestureEvent:
    """One committed gesture.

    event_type  which game action to perform
    confidence  classifier confidence in [0, 1] (1.0 for keyboard)
    timestamp   now() at the moment the classifier committed / emitted
    t_capture   now() when the camera frame that triggered the commit was read;
                optional, but needed for capture-to-display latency
    stages      optional extra stage timestamps, e.g.
                {"t_landmarks": ..., "t_inference": ...}
    source      free-form tag such as "keyboard", "velocity", "lstm_v1"
    """

    event_type: EventType
    confidence: float
    timestamp: float
    t_capture: float | None = None
    stages: dict[str, float] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "GestureEvent":
        return cls(
            event_type=EventType.parse(d["event_type"]),
            confidence=float(d.get("confidence", 1.0)),
            timestamp=float(d["timestamp"]),
            t_capture=None if d.get("t_capture") is None else float(d["t_capture"]),
            stages={k: float(v) for k, v in (d.get("stages") or {}).items()},
            source=str(d.get("source", "")),
        )

    @classmethod
    def from_json(cls, s: str | bytes) -> "GestureEvent":
        return cls.from_dict(json.loads(s))


class EventSender:
    """Fire-and-forget UDP sender used by the classifier process."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, source: str = ""):
        self.addr = (host, port)
        self.source = source
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(
        self,
        event_type: EventType | str,
        confidence: float,
        t_capture: float | None = None,
        timestamp: float | None = None,
        **stages: float,
    ) -> GestureEvent:
        if not isinstance(event_type, EventType):
            event_type = EventType.parse(event_type)
        event = GestureEvent(
            event_type=event_type,
            confidence=float(confidence),
            timestamp=now() if timestamp is None else timestamp,
            t_capture=t_capture,
            stages=dict(stages),
            source=self.source,
        )
        self.sock.sendto(event.to_json().encode(), self.addr)
        return event

    def close(self) -> None:
        self.sock.close()


class EventReceiver:
    """Non-blocking UDP receiver polled once per game frame."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.malformed = 0

    def poll(self) -> list[tuple[GestureEvent, float]]:
        """Drain every pending datagram. Returns (event, t_recv) pairs in arrival order."""
        out = []
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except BlockingIOError:
                break
            t_recv = now()
            try:
                out.append((GestureEvent.from_json(data), t_recv))
            except (ValueError, KeyError, TypeError):
                self.malformed += 1
        return out

    def close(self) -> None:
        self.sock.close()

"""Gesture events -> Subway Surfers (transfer arm).

Listens on the same UDP interface as the runner, so any classifier that drives
the runner drives Subway Surfers unchanged. Two output backends:

  keys  arrow keys via pynput, for the browser version (focus the game window).
        macOS: allow your terminal / VS Code under System Settings > Privacy &
        Security > Accessibility, or the key presses are silently dropped.
  adb   touch swipes on an Android emulator or device (BlueStacks, Waydroid,
        or a phone with USB debugging) via `adb shell input swipe`.

    python -m integration.subway_bridge --backend keys            # then start a classifier
    python -m integration.subway_bridge --backend adb --selftest left,right,jump,roll

--selftest skips UDP and fires the listed actions one per --interval seconds:
the week-4 spike check that a synthetic input moves the character.

The game exposes no state, so the log records only what was dispatched and
when (t_recv -> t_dispatched); score is read off the screen by hand.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

from common.events import DEFAULT_PORT, EventReceiver, EventType, now

LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "subway"


class KeyBackend:
    def __init__(self, hold_s: float):
        from pynput.keyboard import Controller, Key

        self.kb = Controller()
        self.hold_s = hold_s  # games can miss instant taps
        self.keys = {EventType.LEFT: Key.left, EventType.RIGHT: Key.right,
                     EventType.JUMP: Key.up, EventType.ROLL: Key.down}

    def dispatch(self, action: EventType) -> None:
        key = self.keys[action]
        self.kb.press(key)
        time.sleep(self.hold_s)
        self.kb.release(key)


class AdbBackend:
    def __init__(self, serial: str | None, swipe_ms: int):
        self.base = ["adb"] + (["-s", serial] if serial else [])
        self.swipe_ms = swipe_ms
        out = subprocess.run(self.base + ["shell", "wm", "size"], capture_output=True,
                             text=True, timeout=5, check=True).stdout
        # "Physical size: 1080x2400" (an "Override size" line, if present, comes last)
        w, h = map(int, out.strip().splitlines()[-1].split(":")[-1].strip().split("x"))
        cx, cy, dx, dy = w // 2, h // 2, w // 4, h // 5
        self.swipes = {EventType.LEFT: (cx + dx, cy, cx - dx, cy),
                       EventType.RIGHT: (cx - dx, cy, cx + dx, cy),
                       EventType.JUMP: (cx, cy + dy, cx, cy - dy),
                       EventType.ROLL: (cx, cy - dy, cx, cy + dy)}

    def dispatch(self, action: EventType) -> None:
        coords = [str(c) for c in self.swipes[action]]
        subprocess.run(self.base + ["shell", "input", "swipe", *coords, str(self.swipe_ms)],
                       check=True, timeout=5)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m integration.subway_bridge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("keys", "adb"), default="keys")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--hold", type=float, default=0.06, help="keys: seconds to hold each key")
    p.add_argument("--serial", help="adb: device serial, if more than one is attached")
    p.add_argument("--swipe-ms", type=int, default=80, help="adb: swipe duration")
    p.add_argument("--selftest", help="comma-separated actions to fire without a classifier")
    p.add_argument("--interval", type=float, default=1.5, help="selftest: seconds between actions")
    p.add_argument("--participant", default="dev")
    p.add_argument("--log-dir", type=Path, default=LOG_DIR)
    a = p.parse_args(argv)

    backend = KeyBackend(a.hold) if a.backend == "keys" else AdbBackend(a.serial, a.swipe_ms)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = a.log_dir / f"{stamp}_{a.participant}_subway_{a.backend}.jsonl"
    log = open(log_path, "w")
    log.write(json.dumps({"type": "session_start", "backend": a.backend, "participant": a.participant,
                          "min_confidence": a.min_confidence,
                          "wall_time": dt.datetime.now().isoformat(timespec="seconds")}) + "\n")

    def handle(action: EventType, rec: dict) -> None:
        if rec.get("accepted", True):
            backend.dispatch(action)
            rec["t_dispatched"] = now()
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"{action.value:5s} {'sent' if 't_dispatched' in rec else 'ignored (low confidence)'}")

    try:
        if a.selftest:
            print(f"focus the game window; first action in 3 s, then every {a.interval} s")
            time.sleep(3)
            for name in a.selftest.split(","):
                action = EventType.parse(name)
                handle(action, {"type": "selftest", "action": action.value, "t_recv": now()})
                time.sleep(a.interval)
        else:
            receiver = EventReceiver(port=a.port)
            print(f"listening on 127.0.0.1:{a.port}; Ctrl-C to stop")
            while True:
                for ev, t_recv in receiver.poll():
                    ok = ev.confidence >= a.min_confidence
                    handle(ev.event_type, {"type": "input", "event": ev.to_dict(), "t_recv": t_recv,
                                           "accepted": ok})
                time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        log.close()
        print(f"log: {log_path}")


if __name__ == "__main__":
    main()

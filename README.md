# Early Gesture Classification for Low-Latency Camera-Based Game Control

CS 4701 project: Sohan Madishetty (skm254), Sanya Mahajan (sm2588), Tanya Aravind (ta374).

A webcam gesture layer (MediaPipe landmarks -> temporal model -> commit policy) that drives a
three-lane endless runner we built for the study, with a transfer arm onto Subway Surfers.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest          # game rules, determinism, event interface, log -> replay
```

## Layout

| Path | What |
|---|---|
| `common/events.py` | The shared event interface: classifier sends `(event_type, confidence, timestamp)` as JSON over UDP `127.0.0.1:5005`. The only module both sides import. |
| `common/send_test_events.py` | Fake classifier: send scripted, random, or typed events to a running game. |
| `game/` | The runner: deterministic seeded world, pygame renderer, keyboard/gesture/autopilot input, JSONL logging, replay, latency report, latency-budget sweep. |
| `integration/threshold_bridge.py` | Webcam -> MediaPipe Hands -> velocity threshold -> events. Go/no-go latency check and the hand-tuned baseline. |
| `integration/subway_bridge.py` | Events -> Subway Surfers (browser arrow keys or Android `adb` swipes). |
| `tests/` | pytest suite. |

## Quick start

```bash
python -m game                                   # keyboard: arrows/WASD, P pause, R restart, Esc quit
python -m game --input gesture                   # terminal 1: wait for classifier events
python -m integration.threshold_bridge           # terminal 2: webcam velocity baseline
python -m game.latency_report logs/*.jsonl       # where the milliseconds go
python -m game.replay logs/<session>.jsonl       # verify a session reproduces exactly
python -m game.sweep                             # simulated latency budget
```

## Sending events from a classifier

```python
from common.events import EventSender, now
sender = EventSender(source="lstm_v1")
t_cap = now()                      # right after cap.read()
...                                # landmarks, features, model
sender.send("left", confidence=0.93, t_capture=t_cap, t_landmarks=..., t_inference=...)
```

Event names: `left`, `right`, `jump`, `roll` (aliases `up`, `down`, `swipe_left`, ... also accepted).
Timestamps must come from `common.events.now()` so the game can decompose latency.

# Event interface (classifier -> game)

The one contract between the perception/model track and the environment track.
Everything that consumes gestures (the runner, the scrolling prototype, the
Subway Surfers bridge) listens for the same messages, so a classifier written
once drives all three.

## Transport

JSON over UDP to `127.0.0.1:5005` (override with `--port` on both sides). One
datagram per committed gesture. Fire and forget: the classifier never waits on
the game. Malformed packets are counted (`malformed_packets` in the session
summary) and dropped.

## Message

```json
{
  "event_type": "left",
  "confidence": 0.93,
  "timestamp": 58461.5167,
  "t_capture": 58461.3921,
  "stages": {"t_landmarks": 58461.4310, "t_inference": 58461.5102},
  "source": "lstm_v1"
}
```

| Field | Required | Meaning |
|---|---|---|
| `event_type` | yes | `left`, `right`, `jump`, `roll`. Aliases accepted: `up`/`down`, `swipe_left`/..., `l`/`r`/`u`/`d`. |
| `confidence` | no (default 1.0) | Model confidence in [0, 1]. The game drops events below `--min-confidence` but still logs them. |
| `timestamp` | yes | `now()` when the commit policy fired. |
| `t_capture` | strongly recommended | `now()` right after `cap.read()` returned the frame that triggered the commit. Without it the latency report starts at `timestamp`. |
| `stages` | no | Any extra `t_*` timestamps; they become segments in the latency report, ordered by time. |
| `source` | no | Free-form tag (`velocity_threshold`, `lstm_v1`, ...). The latency report groups by it. |

**All timestamps must come from `common.events.now()`** (`time.monotonic()`).
On macOS and Linux that clock is shared across processes, which is what lets the
game subtract the classifier's timestamps from its own.

## Sending (Python)

```python
from common.events import EventSender, now

sender = EventSender(source="lstm_v1")
while True:
    ok, frame = cap.read()
    t_capture = now()
    landmarks = hands.process(frame); t_landmarks = now()
    probs = model(features(landmarks)); t_inference = now()
    if commit_policy(probs):
        sender.send(label, confidence=probs.max(), t_capture=t_capture,
                    t_landmarks=t_landmarks, t_inference=t_inference)
```

`integration/threshold_bridge.py` is a complete working example (webcam ->
MediaPipe -> velocity threshold -> events).

## Commit-policy obligations

The game applies every event it receives, so the classifier side is responsible
for emitting each gesture **once**: debounce or cool down after a commit, and
emit nothing for the null/idle class. A double-fired swipe moves two lanes.

## Testing without a classifier

```bash
python -m game --input gesture
python -m common.send_test_events --interactive                  # type l/r/u/d
python -m common.send_test_events --random --rate 2 --fake-pipeline-ms 120
```

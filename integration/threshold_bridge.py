"""Webcam -> MediaPipe Hands -> velocity threshold -> game events.

The trivial threshold classifier for the go/no-go latency checkpoint: wire the
simplest possible detector to the game and measure capture-to-display latency
before investing in the learned model. It also serves as the hand-tuned
velocity baseline. The learned classifier replaces it by sending events the
same way (see docs/event_interface.md).

    python -m game --input gesture                  # terminal 1
    python -m integration.threshold_bridge          # terminal 2
    python -m game.latency_report logs/<newest>.jsonl

A swipe fires when the palm center travels more than --thresh hand-lengths in
--window seconds, mostly along one axis. Hand length (wrist to middle-finger
knuckle) normalizes for distance from the camera.

macOS: the first run asks for camera permission for your terminal / VS Code.
"""

from __future__ import annotations

import argparse
import math
from collections import deque

import cv2
import mediapipe as mp

from common.events import DEFAULT_PORT, EventSender, EventType, now

PALM = (0, 5, 9, 13, 17)  # wrist + finger bases: a stable palm center


def detect(track: deque, thresh: float) -> tuple[EventType, float] | None:
    """track holds (t, x, y, hand_len) in mirrored image coords (x right, y down)."""
    if len(track) < 3:
        return None
    _, x0, y0, _ = track[0]
    _, x1, y1, _ = track[-1]
    scale = sum(h for *_, h in track) / len(track)
    dx, dy = (x1 - x0) / scale, (y1 - y0) / scale
    mag = math.hypot(dx, dy)
    if mag < thresh:
        return None
    if abs(dx) > 1.5 * abs(dy):
        kind = EventType.RIGHT if dx > 0 else EventType.LEFT
    elif abs(dy) > 1.5 * abs(dx):
        kind = EventType.ROLL if dy > 0 else EventType.JUMP
    else:
        return None  # diagonal: ambiguous, don't fire
    return kind, min(1.0, mag / (2 * thresh))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m integration.threshold_bridge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--thresh", type=float, default=1.2, help="hand-lengths of travel")
    p.add_argument("--window", type=float, default=0.25, help="seconds")
    p.add_argument("--cooldown", type=float, default=0.4, help="seconds after a swipe")
    p.add_argument("--no-preview", action="store_true")
    a = p.parse_args(argv)

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit("could not open camera (check camera permission for your terminal)")
    sender = EventSender(port=a.port, source="velocity_threshold")
    hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0,
                                     min_detection_confidence=0.6, min_tracking_confidence=0.5)
    track: deque = deque()
    cooldown_until = 0.0
    last = None
    print(f"sending to 127.0.0.1:{a.port}; press q in the preview window (or Ctrl-C) to quit")
    try:
        while True:
            ok, frame = cap.read()
            t_capture = now()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror: your left is image left
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t_landmarks = now()

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark
                cx = sum(lm[i].x for i in PALM) / len(PALM)
                cy = sum(lm[i].y for i in PALM) / len(PALM)
                hand_len = math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y) or 1e-3
                track.append((t_capture, cx, cy, hand_len))
            else:
                track.clear()
            while track and t_capture - track[0][0] > a.window:
                track.popleft()

            hit = detect(track, a.thresh) if t_capture >= cooldown_until else None
            t_inference = now()
            if hit:
                kind, conf = hit
                sender.send(kind, conf, t_capture=t_capture,
                            t_landmarks=t_landmarks, t_inference=t_inference)
                cooldown_until = t_capture + a.cooldown
                track.clear()
                last = (kind.value, conf, t_capture)
                print(f"{kind.value:5s} conf={conf:.2f}  pipeline {1000 * (now() - t_capture):.0f} ms")

            if not a.no_preview:
                if result.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
                if last and t_capture - last[2] < 0.6:
                    cv2.putText(frame, f"{last[0]} {last[1]:.2f}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (40, 200, 40), 3)
                cv2.imshow("threshold bridge (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        hands.close()
        sender.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

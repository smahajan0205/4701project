import numpy as np


def detect_gesture(wrist_positions, timestamps, threshold=0.8):
    """
    Detect swipe direction using wrist velocity.

    wrist_positions: array of shape (N, 2)
    timestamps: array of timestamps in milliseconds
    threshold: minimum velocity in normalized coordinates/sec
    """

    if len(wrist_positions) < 2:
        return "idle"

    positions = np.asarray(wrist_positions, dtype=float)
    times = np.asarray(timestamps, dtype=float)

    dt = (times[-1] - times[0]) / 1000.0

    if dt <= 0:
        return "idle"

    displacement = positions[-1] - positions[0]
    velocity = displacement / dt

    vx, vy = velocity

    if abs(vx) > abs(vy) and abs(vx) > threshold:
        return "right" if vx > 0 else "left"

    if abs(vy) > threshold:
        return "down" if vy > 0 else "up"

    return "idle"

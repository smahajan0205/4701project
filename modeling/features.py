"""Feature extraction from normalized hand landmark sequences.

Inputs (one sequence — a full clip or one sliding window):
    landmarks  [T, 21, 3]  normalized landmarks (wrist at origin, scaled by
                           wrist-to-middle-MCP distance)
    wrist_xy   [T, 2]      raw wrist position in image coordinates (0-1)
    hand_scale [T]         raw wrist-to-middle-MCP distance in image
                           coordinates (the divisor used by normalize_landmarks)
Output: [T, FEAT_DIM] per-frame feature vectors.

Features per frame (FEAT_DIM = 130):
    positions   21 * 3 = 63  — flattened normalized landmark coordinates
    velocity    21 * 3 = 63  — inter-frame delta of the above (zero for frame 0)
    wrist disp       2       — wrist displacement from the sequence's first
                               frame, in hand-lengths
    wrist vel        2       — inter-frame wrist delta, in hand-lengths

The normalized landmarks are wrist-centred, so they carry hand shape but not
hand translation; the wrist terms restore the translation that distinguishes
left/right swipes (and jump/roll).  Dividing by the median hand scale keeps
them invariant to camera distance without amplifying per-frame jitter.
"""

from __future__ import annotations

import numpy as np

FEAT_DIM = 130


def extract_features(
    landmarks: np.ndarray,
    wrist_xy: np.ndarray,
    hand_scale: np.ndarray,
) -> np.ndarray:
    """Convert one sequence to [T, FEAT_DIM] feature vectors."""
    T = len(landmarks)
    positions = landmarks.reshape(T, -1).astype(np.float32)  # [T, 63]
    velocity = np.zeros_like(positions)
    if T > 1:
        velocity[1:] = positions[1:] - positions[:-1]

    scale = max(float(np.median(hand_scale)), 1e-6)
    wrist = wrist_xy.astype(np.float32) / scale              # [T, 2]
    wrist_disp = wrist - wrist[0]
    wrist_vel = np.zeros_like(wrist)
    if T > 1:
        wrist_vel[1:] = wrist[1:] - wrist[:-1]

    return np.concatenate([positions, velocity, wrist_disp, wrist_vel], axis=-1)

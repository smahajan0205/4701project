"""Feature extraction from normalized hand landmark sequences.

Input: [T, 21, 3] normalized landmarks (wrist at origin, scaled by hand length).
Output: [T, FEAT_DIM] per-frame feature vectors.

Features per frame (FEAT_DIM = 126):
    positions  21 * 3 = 63  — flattened normalized landmark coordinates
    velocity   21 * 3 = 63  — inter-frame delta (zero for the first frame)
"""

from __future__ import annotations

import numpy as np

FEAT_DIM = 126


def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """Convert [T, 21, 3] landmarks to [T, FEAT_DIM] feature vectors."""
    T = len(landmarks)
    positions = landmarks.reshape(T, -1).astype(np.float32)  # [T, 63]
    velocity = np.zeros_like(positions)
    if T > 1:
        velocity[1:] = positions[1:] - positions[:-1]
    return np.concatenate([positions, velocity], axis=-1)    # [T, 126]

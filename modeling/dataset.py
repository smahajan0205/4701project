"""Gesture dataset: loading recorded clips, windowing, and LOUO splits.

Reads the per-clip CSVs written by perception/record_gestures.py, named
{user}_{gesture}_{rep:03d}.csv, one row per frame with columns:
    timestamp_ms, wrist_x, wrist_y, hand_scale, hand_detected,
    landmark_{0..20}_{x,y,z}

Frames with hand_detected == 0 are dropped.  The recorder's gesture names
are mapped onto the game's event names (up -> jump, down -> roll), matching
common/events.py.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from modeling.features import extract_features

GESTURE_CLASSES = ["left", "right", "jump", "roll", "idle"]
CLASS_TO_IDX = {c: i for i, c in enumerate(GESTURE_CLASSES)}
RECORDER_TO_CLASS = {"left": "left", "right": "right", "up": "jump",
                     "down": "roll", "idle": "idle"}
LANDMARK_COLS = [f"landmark_{i}_{a}" for i in range(21) for a in "xyz"]
MIN_FRAMES = 2


def load_clip(path: str | Path) -> dict | None:
    """Load one recorder CSV.  Returns None if fewer than MIN_FRAMES frames
    had a detected hand."""
    path = Path(path)
    user_id, gesture, _rep = path.stem.rsplit("_", 2)
    if gesture not in RECORDER_TO_CLASS:
        raise ValueError(f"{path.name}: unknown gesture {gesture!r}")

    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["hand_detected"] == "1"]
    if len(rows) < MIN_FRAMES:
        return None

    def col(names):
        return np.array([[float(r[n]) for n in names] for r in rows], dtype=np.float32)

    return {
        "landmarks": col(LANDMARK_COLS).reshape(-1, 21, 3),
        "wrist_xy": col(["wrist_x", "wrist_y"]),
        "hand_scale": col(["hand_scale"])[:, 0],
        "timestamps_ms": col(["timestamp_ms"])[:, 0],
        "label": RECORDER_TO_CLASS[gesture],
        "user_id": user_id,
        "path": path,
    }


def load_recordings(data_dir: str | Path) -> list[dict]:
    """Load all recorder CSVs from *data_dir*, skipping clips with no hand.

    Returns a list of dicts with keys: landmarks [T,21,3], wrist_xy [T,2],
    hand_scale [T], timestamps_ms [T], label, user_id, path.
    """
    records, empty = [], []
    for path in sorted(Path(data_dir).glob("*.csv")):
        rec = load_clip(path)
        if rec is None:
            empty.append(path.name)
        else:
            records.append(rec)
    if empty:
        warnings.warn(f"skipped {len(empty)} clip(s) with < {MIN_FRAMES} detected "
                      f"frames: {', '.join(empty[:5])}{' ...' if len(empty) > 5 else ''}")
    return records


def louo_splits(records: list[dict]) -> Iterator[tuple[list[dict], list[dict], str]]:
    """Leave-one-user-out cross-validation splits.

    Yields (train_records, test_records, held_out_user_id) for each user.
    Holding out by user avoids the inflated accuracy that results when the
    same person's clips appear in both train and test.
    """
    for user in sorted({r["user_id"] for r in records}):
        train = [r for r in records if r["user_id"] != user]
        test = [r for r in records if r["user_id"] == user]
        yield train, test, user


class GestureDataset(Dataset):
    """PyTorch Dataset of fixed-length landmark windows.

    Each item is (features_tensor, label_idx) where features_tensor has
    shape [window, feat_dim].  Long recordings are sliced into overlapping
    windows; short ones are zero-padded on the left, so that models reading
    the final timestep (GestureLSTM) see real frames rather than padding.

    Features are computed per window, not per recording, so window-relative
    terms (wrist displacement) match what live inference will see.
    """

    def __init__(
        self,
        records: list[dict],
        window: int = 20,
        stride: int = 5,
        feature_fn=extract_features,
    ):
        """
        Args:
            records   : output of load_recordings()
            window    : number of frames per sample
            stride    : step between sliding windows (< window → overlap)
            feature_fn: callable (landmarks, wrist_xy, hand_scale) -> [T, F]
        """
        self.window = window
        self.feature_fn = feature_fn
        self._items: list[tuple[np.ndarray, int]] = []

        for rec in records:
            lm, wr, sc = rec["landmarks"], rec["wrist_xy"], rec["hand_scale"]
            label_idx = CLASS_TO_IDX[rec["label"]]
            T = len(lm)

            if T < window:
                feats = feature_fn(lm, wr, sc)
                pad = np.zeros((window - T, feats.shape[1]), dtype=np.float32)
                self._items.append((np.concatenate([pad, feats], axis=0), label_idx))
            else:
                for s in range(0, T - window + 1, stride):
                    e = s + window
                    self._items.append((feature_fn(lm[s:e], wr[s:e], sc[s:e]), label_idx))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        feats, label = self._items[idx]
        return torch.tensor(feats, dtype=torch.float32), label

"""Gesture dataset: loading .npz recordings, windowing, and LOUO splits.

Expected .npz file format (produced by perception/record_gestures.py):
    landmarks     : float32 [T, 21, 3]  - normalized hand landmarks per frame
    label         : str                 - one of GESTURE_CLASSES
    user_id       : str
    timestamps_ms : float32 [T]         - milliseconds from session start
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

GESTURE_CLASSES = ["left", "right", "jump", "roll", "idle"]
CLASS_TO_IDX = {c: i for i, c in enumerate(GESTURE_CLASSES)}


def load_recordings(data_dir: str | Path) -> list[dict]:
    """Load all .npz gesture recordings from *data_dir*.

    Returns a list of dicts with keys: landmarks, label, user_id,
    timestamps_ms, path.
    """
    records = []
    for path in sorted(Path(data_dir).glob("*.npz")):
        d = np.load(path, allow_pickle=True)
        records.append({
            "landmarks": d["landmarks"].astype(np.float32),
            "label": str(d["label"]),
            "user_id": str(d["user_id"]),
            "timestamps_ms": d["timestamps_ms"].astype(np.float32),
            "path": path,
        })
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
    windows; short ones are zero-padded on the right.
    """

    def __init__(
        self,
        records: list[dict],
        window: int = 20,
        stride: int = 5,
        feature_fn=None,
    ):
        """
        Args:
            records   : output of load_recordings()
            window    : number of frames per sample
            stride    : step between sliding windows (< window → overlap)
            feature_fn: callable [[T,21,3]] -> [[T,F]]; defaults to flatten
        """
        self.window = window
        self.feature_fn = feature_fn
        self._items: list[tuple[np.ndarray, int]] = []

        for rec in records:
            lm = rec["landmarks"]          # [T, 21, 3]
            label_idx = CLASS_TO_IDX[rec["label"]]
            feats = feature_fn(lm) if feature_fn else lm.reshape(len(lm), -1)
            T, F = feats.shape

            if T < window:
                pad = np.zeros((window - T, F), dtype=np.float32)
                self._items.append((np.concatenate([feats, pad], axis=0), label_idx))
            else:
                for start in range(0, T - window + 1, stride):
                    self._items.append((feats[start: start + window], label_idx))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        feats, label = self._items[idx]
        return torch.tensor(feats, dtype=torch.float32), label

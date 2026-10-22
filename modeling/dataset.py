"""Gesture dataset: loading .npz recordings, windowing, and LOUO splits.

Expected .npz file format (produced by perception/record_gestures.py):
    landmarks     : float32 [T, 21, 3]  - normalized hand landmarks per frame
    wrist_xy      : float32 [T, 2]      - raw wrist (landmark 0) x, y in image coords
    hand_scale    : float32 [T]         - raw wrist-to-middle-MCP distance in image coords
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

from modeling.features import extract_features

GESTURE_CLASSES = ["left", "right", "jump", "roll", "idle"]
CLASS_TO_IDX = {c: i for i, c in enumerate(GESTURE_CLASSES)}
REQUIRED_FIELDS = ("landmarks", "wrist_xy", "hand_scale", "label", "user_id", "timestamps_ms")


def load_recordings(data_dir: str | Path) -> list[dict]:
    """Load all .npz gesture recordings from *data_dir*.

    Returns a list of dicts with keys: landmarks, wrist_xy, hand_scale,
    label, user_id, timestamps_ms, path.
    """
    records = []
    for path in sorted(Path(data_dir).glob("*.npz")):
        d = np.load(path, allow_pickle=True)
        missing = [k for k in REQUIRED_FIELDS if k not in d.files]
        if missing:
            raise ValueError(f"{path.name} is missing fields {missing}")
        records.append({
            "landmarks": d["landmarks"].astype(np.float32),
            "wrist_xy": d["wrist_xy"].astype(np.float32),
            "hand_scale": d["hand_scale"].astype(np.float32),
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

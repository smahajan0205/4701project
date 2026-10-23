"""Smoke-tests for the modeling module (dataset, features, models, evaluate)."""

import numpy as np
import pytest
import torch

from modeling.dataset import (
    CLASS_TO_IDX,
    GESTURE_CLASSES,
    LANDMARK_COLS,
    GestureDataset,
    load_recordings,
    louo_splits,
)
from modeling.evaluate import (
    accuracy_vs_fraction,
    confusion_matrix,
    per_class_f1,
    predict_all,
)
from modeling.features import FEAT_DIM, extract_features
from modeling.models import BaselineMLP, GestureLSTM, NUM_CLASSES


def _make_record(user: str, label: str, T: int = 25) -> dict:
    return {
        "landmarks": np.random.randn(T, 21, 3).astype(np.float32),
        "wrist_xy": np.random.rand(T, 2).astype(np.float32),
        "hand_scale": np.full(T, 0.1, dtype=np.float32),
        "label": label,
        "user_id": user,
        "timestamps_ms": np.arange(T, dtype=np.float32) * 33.0,
        "path": None,
    }


# ----------------------------------------------------------------- features

def _static_wrist(T: int, x: float = 0.5, y: float = 0.5, scale: float = 0.1):
    wrist = np.tile(np.array([[x, y]], dtype=np.float32), (T, 1))
    return wrist, np.full(T, scale, dtype=np.float32)


def test_extract_features_shape():
    lm = np.ones((30, 21, 3), dtype=np.float32)
    feats = extract_features(lm, *_static_wrist(30))
    assert feats.shape == (30, FEAT_DIM)


def test_velocity_zero_for_static_hand():
    lm = np.ones((10, 21, 3), dtype=np.float32)
    feats = extract_features(lm, *_static_wrist(10))
    assert np.allclose(feats[1:, 63:], 0.0)  # landmark vel, wrist disp, wrist vel


def test_wrist_features_separate_left_and_right_swipes():
    # Identical hand shape; only the wrist translates. Wrist-centred landmarks
    # alone cannot tell these apart, the wrist features must.
    T = 10
    lm = np.ones((T, 21, 3), dtype=np.float32)
    xs = np.linspace(0.3, 0.7, T, dtype=np.float32)
    scale = np.full(T, 0.1, dtype=np.float32)
    right = extract_features(lm, np.stack([xs, np.full(T, 0.5)], 1), scale)
    left = extract_features(lm, np.stack([xs[::-1], np.full(T, 0.5)], 1), scale)
    assert np.allclose(right[:, :126], left[:, :126])
    assert right[-1, 126] > 0 > left[-1, 126]          # x displacement sign
    assert np.isclose(right[-1, 126], 4.0)             # 0.4 image units / 0.1 hand length


def test_wrist_features_invariant_to_camera_distance():
    T = 10
    lm = np.ones((T, 21, 3), dtype=np.float32)
    xs = np.linspace(0.3, 0.7, T, dtype=np.float32)
    near = extract_features(lm, np.stack([xs, np.full(T, 0.5)], 1),
                            np.full(T, 0.2, dtype=np.float32))
    # twice as far: half the image-space motion and half the hand size
    far = extract_features(lm, np.stack([xs / 2, np.full(T, 0.25)], 1),
                           np.full(T, 0.1, dtype=np.float32))
    assert np.allclose(near, far, atol=1e-5)


# ----------------------------------------------------------------- dataset

def test_gesture_dataset_window_count():
    records = [_make_record("u1", "left", T=30)]
    ds = GestureDataset(records, window=20, stride=5, feature_fn=extract_features)
    assert len(ds) == 3  # windows starting at 0, 5, 10


def test_gesture_dataset_short_recording_padded():
    records = [_make_record("u1", "right", T=10)]
    ds = GestureDataset(records, window=20, stride=5, feature_fn=extract_features)
    assert len(ds) == 1
    x, _ = ds[0]
    assert x.shape == (20, FEAT_DIM)
    # padding goes on the left so the final timestep is real data
    assert torch.all(x[:10] == 0)
    assert torch.any(x[-1] != 0)


def test_gesture_dataset_label_index():
    for label in GESTURE_CLASSES:
        records = [_make_record("u1", label)]
        ds = GestureDataset(records, window=20, feature_fn=extract_features)
        _, y = ds[0]
        assert y == CLASS_TO_IDX[label]


def _write_clip(path, detected):
    import csv
    header = ["timestamp_ms", "wrist_x", "wrist_y", "hand_scale", "hand_detected"] + LANDMARK_COLS
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t, d in enumerate(detected):
            if d:
                w.writerow([t * 33, 0.5 + 0.01 * t, 0.5, 0.1, 1] + [0.0] * 63)
            else:
                w.writerow([t * 33, "", "", "", 0] + [""] * 63)


def test_load_recordings_reads_recorder_csvs(tmp_path):
    _write_clip(tmp_path / "sanya_up_001.csv", [0, 1, 1, 0, 1])
    _write_clip(tmp_path / "sohan_left_002.csv", [1, 1, 1])
    _write_clip(tmp_path / "tanya_down_001.csv", [0, 0, 0])  # no hand: skipped
    with pytest.warns(UserWarning, match="tanya_down_001"):
        records = load_recordings(tmp_path)
    by_user = {r["user_id"]: r for r in records}
    assert set(by_user) == {"sanya", "sohan"}
    up = by_user["sanya"]
    assert up["label"] == "jump"                         # recorder name -> game event
    assert up["landmarks"].shape == (3, 21, 3)           # undetected frames dropped
    assert up["wrist_xy"].shape == (3, 2) and up["hand_scale"].shape == (3,)
    assert np.allclose(up["timestamps_ms"], [33, 66, 132])


def test_louo_splits_hold_out_every_user():
    records = (
        [_make_record("u1", c) for c in GESTURE_CLASSES]
        + [_make_record("u2", c) for c in GESTURE_CLASSES]
    )
    splits = list(louo_splits(records))
    assert len(splits) == 2
    for train, test, user in splits:
        assert all(r["user_id"] != user for r in train)
        assert all(r["user_id"] == user for r in test)


def test_split_by_recording_is_disjoint():
    from modeling.train import split_by_recording

    records = [_make_record(f"u{i % 3}", c) for i in range(4) for c in GESTURE_CLASSES]
    for i, r in enumerate(records):
        r["path"] = f"rec_{i}"
    train, val = split_by_recording(records, val_frac=0.2)
    assert len(train) + len(val) == len(records)
    assert len(val) == 4
    assert not {r["path"] for r in train} & {r["path"] for r in val}


# ----------------------------------------------------------------- models

def test_baseline_mlp_output_shape():
    model = BaselineMLP(window=20)
    x = torch.randn(8, 20, FEAT_DIM)
    out = model(x)
    assert out.shape == (8, NUM_CLASSES)


def test_mlp_and_lstm_parameter_budgets_match():
    # The proposal compares models at matched parameter counts.
    n_mlp = sum(p.numel() for p in BaselineMLP(window=20).parameters())
    n_lstm = sum(p.numel() for p in GestureLSTM().parameters())
    assert abs(n_mlp - n_lstm) / n_lstm < 0.01


def test_gesture_lstm_output_shape():
    model = GestureLSTM()
    x = torch.randn(8, 20, FEAT_DIM)
    assert model(x).shape == (8, NUM_CLASSES)


def test_gesture_lstm_forward_partial():
    model = GestureLSTM()
    x = torch.randn(4, 20, FEAT_DIM)
    for frac in (0.25, 0.5, 0.75, 1.0):
        out = model.forward_partial(x, frac)
        assert out.shape == (4, NUM_CLASSES)


# --------------------------------------------------------------- evaluation

def test_confusion_matrix_shape_and_diagonal():
    y_true = np.arange(NUM_CLASSES)
    y_pred = np.arange(NUM_CLASSES)
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
    assert np.trace(cm) == NUM_CLASSES


def test_per_class_f1_perfect():
    y = np.arange(NUM_CLASSES)
    cm = confusion_matrix(y, y)
    f1s = per_class_f1(cm)
    assert all(abs(f - 1.0) < 1e-6 for f in f1s)


def test_accuracy_vs_fraction_keys():
    records = [_make_record("u1", c, T=T) for c, T in zip(GESTURE_CLASSES, (8, 15, 25, 40, 60))]
    model = GestureLSTM()
    result = accuracy_vs_fraction(model, records, torch.device("cpu"))
    assert set(result.keys()) == {0.25, 0.5, 0.75, 1.0}
    assert all(0.0 <= v <= 1.0 for v in result.values())

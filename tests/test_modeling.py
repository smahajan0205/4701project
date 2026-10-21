"""Smoke-tests for the modeling module (dataset, features, models, evaluate)."""

import numpy as np
import pytest
import torch

from modeling.dataset import (
    CLASS_TO_IDX,
    GESTURE_CLASSES,
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
        "label": label,
        "user_id": user,
        "timestamps_ms": np.arange(T, dtype=np.float32) * 33.0,
        "path": None,
    }


# ----------------------------------------------------------------- features

def test_extract_features_shape():
    lm = np.ones((30, 21, 3), dtype=np.float32)
    feats = extract_features(lm)
    assert feats.shape == (30, FEAT_DIM)


def test_velocity_zero_for_static_hand():
    lm = np.ones((10, 21, 3), dtype=np.float32)
    feats = extract_features(lm)
    velocity = feats[:, 63:]
    assert np.allclose(velocity[1:], 0.0)


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


def test_gesture_dataset_label_index():
    for label in GESTURE_CLASSES:
        records = [_make_record("u1", label)]
        ds = GestureDataset(records, window=20, feature_fn=extract_features)
        _, y = ds[0]
        assert y == CLASS_TO_IDX[label]


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


# ----------------------------------------------------------------- models

def test_baseline_mlp_output_shape():
    model = BaselineMLP(window=20)
    x = torch.randn(8, 20, FEAT_DIM)
    out = model(x)
    assert out.shape == (8, NUM_CLASSES)


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
    from torch.utils.data import DataLoader

    records = [_make_record("u1", c) for c in GESTURE_CLASSES]
    ds = GestureDataset(records, window=20, feature_fn=extract_features)
    loader = DataLoader(ds, batch_size=4)
    model = GestureLSTM()
    result = accuracy_vs_fraction(model, loader, torch.device("cpu"))
    assert set(result.keys()) == {0.25, 0.5, 0.75, 1.0}
    assert all(0.0 <= v <= 1.0 for v in result.values())

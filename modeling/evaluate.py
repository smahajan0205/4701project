"""Evaluation utilities for gesture classifiers.

Provides:
    predict_all()           — collect all predictions from a DataLoader
    confusion_matrix()      — compute [C, C] confusion matrix
    per_class_f1()          — per-class F1 scores from confusion matrix
    accuracy_vs_fraction()  — LSTM accuracy at 25/50/75/100% of each gesture clip
    run_louo()              — full leave-one-user-out evaluation loop

CLI usage:
    python -m modeling.evaluate --data data/pilot --checkpoint checkpoints/lstm_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from modeling.dataset import (
    CLASS_TO_IDX,
    GESTURE_CLASSES,
    GestureDataset,
    load_recordings,
    louo_splits,
)
from modeling.features import extract_features
from modeling.models import GestureLSTM

NUM_CLASSES = len(GESTURE_CLASSES)
FRACTIONS = (0.25, 0.50, 0.75, 1.00)


@torch.no_grad()
def predict_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) arrays over the entire loader."""
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        preds = model(x).argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y if isinstance(y, list) else y.numpy())
    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return a [NUM_CLASSES, NUM_CLASSES] confusion matrix."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def per_class_f1(cm: np.ndarray) -> list[float]:
    """Compute per-class F1 from a confusion matrix."""
    f1s = []
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return f1s


@torch.no_grad()
def accuracy_vs_fraction(
    model: GestureLSTM,
    records: list[dict],
    device: torch.device,
    fractions: tuple[float, ...] = FRACTIONS,
    feature_fn=extract_features,
) -> dict[float, float]:
    """Accuracy of the LSTM when shown only the first *frac* of each clip.

    Each recording is one segmented gesture starting at onset, so the prefix
    is measured from the start of the gesture rather than from an arbitrary
    sliding-window offset.  Sweeping fractions traces the
    accuracy-versus-latency curve from the paper.
    """
    model.eval()
    correct = {f: 0 for f in fractions}
    for rec in records:
        feats = feature_fn(rec["landmarks"], rec["wrist_xy"], rec["hand_scale"])
        x = torch.as_tensor(feats, device=device).unsqueeze(0)
        label = CLASS_TO_IDX[rec["label"]]
        for frac in fractions:
            pred = model.forward_partial(x, frac).argmax(1).item()
            correct[frac] += int(pred == label)
    n = len(records)
    return {f: correct[f] / n if n else 0.0 for f in fractions}


def run_louo(
    data_dir: str,
    window: int = 20,
    stride: int = 5,
    epochs: int = 30,
    batch: int = 32,
    lr: float = 1e-3,
) -> list[dict]:
    """Train and evaluate with leave-one-user-out splits.

    Returns per-user result dicts with keys: user, accuracy, f1,
    confusion_matrix, accuracy_vs_fraction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_recordings(data_dir)
    results = []

    for train_recs, test_recs, user in louo_splits(records):
        print(f"\n--- held out: {user} "
              f"(train {len(train_recs)}, test {len(test_recs)}) ---")

        train_ds = GestureDataset(train_recs, window=window, stride=stride,
                                   feature_fn=extract_features)
        test_ds = GestureDataset(test_recs, window=window, stride=1,
                                  feature_fn=extract_features)
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch)

        model = GestureLSTM().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for x, y in train_loader:
                x = x.to(device)
                y = torch.as_tensor(y, dtype=torch.long, device=device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        y_true, y_pred = predict_all(model, test_loader, device)
        cm = confusion_matrix(y_true, y_pred)
        f1s = per_class_f1(cm)
        acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
        frac_accs = accuracy_vs_fraction(model, test_recs, device)

        print(f"  accuracy: {acc:.3f}")
        print("  per-class F1: " + "  ".join(
            f"{GESTURE_CLASSES[i]}={f1s[i]:.3f}" for i in range(NUM_CLASSES)
        ))
        print("  acc vs fraction: " + "  ".join(
            f"{f:.0%}={a:.3f}" for f, a in frac_accs.items()
        ))
        results.append({
            "user": user,
            "accuracy": acc,
            "f1": f1s,
            "confusion_matrix": cm.tolist(),
            "accuracy_vs_fraction": {str(f): a for f, a in frac_accs.items()},
        })

    if results:
        mean_acc = float(np.mean([r["accuracy"] for r in results]))
        print(f"\nLOUO mean accuracy: {mean_acc:.3f}")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True,
                   help="directory of recorder CSVs (perception/record_gestures.py)")
    p.add_argument("--checkpoint",
                   help="path to a saved .pt checkpoint (skips LOUO, runs single eval)")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30,
                   help="epochs per LOUO fold (ignored with --checkpoint)")
    a = p.parse_args(argv)

    if a.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GestureLSTM().to(device)
        model.load_state_dict(torch.load(a.checkpoint, map_location=device))
        records = load_recordings(a.data)
        ds = GestureDataset(records, window=a.window, stride=1,
                             feature_fn=extract_features)
        loader = DataLoader(ds, batch_size=a.batch)
        y_true, y_pred = predict_all(model, loader, device)
        cm = confusion_matrix(y_true, y_pred)
        f1s = per_class_f1(cm)
        acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
        frac_accs = accuracy_vs_fraction(model, records, device)
        print(f"accuracy: {acc:.3f}")
        print("per-class F1:")
        for cls, f in zip(GESTURE_CLASSES, f1s):
            print(f"  {cls:6s}: {f:.3f}")
        print("confusion matrix:")
        _print_cm(cm)
        print("accuracy vs fraction:")
        for frac, a_val in frac_accs.items():
            print(f"  {frac:.0%}: {a_val:.3f}")
    else:
        run_louo(a.data, window=a.window, epochs=a.epochs, batch=a.batch)


def _print_cm(cm: np.ndarray) -> None:
    header = "       " + "  ".join(f"{c[:4]:>5}" for c in GESTURE_CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {GESTURE_CLASSES[i]:5s}  " + "  ".join(f"{v:5d}" for v in row))


if __name__ == "__main__":
    main()

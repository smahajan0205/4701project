"""Training script for gesture classifiers.

Usage:
    python -m modeling.train --data data/gestures --model lstm
    python -m modeling.train --data data/gestures --model mlp --epochs 50

Checkpoints are saved to --out (default: checkpoints/).
The best validation-accuracy checkpoint is named {model}_best.pt.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modeling.dataset import GestureDataset, load_recordings
from modeling.features import extract_features
from modeling.models import BaselineMLP, GestureLSTM


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = n = 0
    for x, y in loader:
        x = x.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = n = 0
    for x, y in loader:
        x = x.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def split_by_recording(records: list[dict], val_frac: float,
                       seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Hold out whole recordings for validation (seeded, deterministic)."""
    if len(records) < 2:
        raise SystemExit("need at least 2 recordings to make a train/val split")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = min(len(shuffled) - 1, max(1, int(val_frac * len(shuffled))))
    return shuffled[n_val:], shuffled[:n_val]


def _build_model(name: str, window: int) -> nn.Module:
    if name == "lstm":
        return GestureLSTM()
    if name == "mlp":
        return BaselineMLP(window=window)
    raise ValueError(f"unknown model: {name!r}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True,
                   help="directory of .npz gesture recordings")
    p.add_argument("--model", choices=["lstm", "mlp"], default="lstm")
    p.add_argument("--window", type=int, default=20,
                   help="frames per sample (default 20 ≈ 0.67 s at 30 fps)")
    p.add_argument("--stride", type=int, default=5,
                   help="step between sliding windows during training")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="fraction of recordings held out for validation")
    p.add_argument("--out", default="checkpoints",
                   help="directory for saved model checkpoints")
    a = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    records = load_recordings(a.data)
    if not records:
        raise SystemExit(f"no .npz files found in {a.data!r} — record gestures first")
    print(f"loaded {len(records)} recordings ({a.data})")

    # Split by recording, not by window: overlapping windows from one clip
    # would otherwise land on both sides and inflate validation accuracy.
    train_recs, val_recs = split_by_recording(records, a.val_frac, seed=42)
    train_ds = GestureDataset(train_recs, window=a.window, stride=a.stride,
                              feature_fn=extract_features)
    val_ds = GestureDataset(val_recs, window=a.window, stride=a.stride,
                            feature_fn=extract_features)
    print(f"train: {len(train_recs)} recordings / {len(train_ds)} windows  "
          f"val: {len(val_recs)} recordings / {len(val_ds)} windows")

    train_loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=a.batch)

    model = _build_model(a.model, a.window).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, a.epochs + 1):
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = _eval_epoch(model, val_loader, criterion, device)
        print(f"epoch {epoch:3d}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val loss={val_loss:.4f} acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / f"{a.model}_best.pt")

    print(f"\nbest val acc: {best_val_acc:.3f}  "
          f"checkpoint: {out_dir}/{a.model}_best.pt")


if __name__ == "__main__":
    main()

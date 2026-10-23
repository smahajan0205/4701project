"""Baseline gesture classifiers.

BaselineMLP  — fixed-window approach: flatten the window, classify with MLP.
               Corresponds to the "always wait k frames" baseline in the paper.
GestureLSTM  — processes the sequence left-to-right; supports partial-window
               inference (forward_partial) for accuracy-vs-latency curves.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modeling.dataset import GESTURE_CLASSES
from modeling.features import FEAT_DIM

NUM_CLASSES = len(GESTURE_CLASSES)


class BaselineMLP(nn.Module):
    """Fixed-window MLP: flatten window × features, then classify.

    hidden=100 matches GestureLSTM's default parameter count (~266k) at the
    default window of 20, so the comparison isolates waiting-for-the-window
    vs. classifying a prefix rather than model size.
    """

    def __init__(self, window: int = 20, feat_dim: int = FEAT_DIM, hidden: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window * feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, window, feat_dim]
        return self.net(x.flatten(1))


class GestureLSTM(nn.Module):
    """Sequence-to-label LSTM for gesture classification and early commitment.

    forward()          — classify using the full window (training / full eval)
    forward_partial()  — classify using only the first *fraction* of each window
                         (accuracy-vs-latency curve)
    """

    def __init__(self, feat_dim: int = FEAT_DIM, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.2 if layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, feat_dim] — classify from last LSTM output
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])

    def forward_partial(self, x: torch.Tensor, fraction: float) -> torch.Tensor:
        """Classify using only the first `fraction` of each sequence (0 < fraction ≤ 1)."""
        T = x.size(1)
        cutoff = max(1, int(T * fraction))
        return self.forward(x[:, :cutoff, :])

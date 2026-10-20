"""Seeded obstacle generation.

Obstacles are generated as a function of distance, never of time, so a given
seed produces the identical course for every participant and every condition,
regardless of frame rate or how they play.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from .config import NUM_LANES, GameConfig


class Kind(str, Enum):
    TRAIN = "train"  # blocks the lane; must change lanes
    LOW = "low"      # low barrier; must jump
    HIGH = "high"    # overhead barrier; must roll


@dataclass
class Obstacle:
    id: int
    row: int
    lane: int
    kind: Kind
    start: float   # track position (m) of the near edge
    length: float

    @property
    def end(self) -> float:
        return self.start + self.length

    def to_dict(self) -> dict:
        return {"id": self.id, "row": self.row, "lane": self.lane,
                "kind": self.kind.value, "start": round(self.start, 3),
                "length": self.length}


# Row patterns and their weights. Each one forces a specific response, so a run
# exercises all four actions.
_PATTERNS = [
    ("single", 4),      # one obstacle of any kind in one lane
    ("two_trains", 3),  # trains in two lanes: must be in the third
    ("wall_low", 2),    # low barriers in every lane: must jump
    ("wall_high", 2),   # overhead barriers in every lane: must roll
    ("mixed", 3),       # train + low + high in some order
]


class ObstacleGenerator:
    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.next_pos = cfg.first_row_m
        self.next_row = 0
        self.next_id = 0

    def generate_until(self, position: float) -> list[Obstacle]:
        """Return every new row whose near edge is at or before `position`."""
        out: list[Obstacle] = []
        while self.next_pos <= position:
            out.extend(self._make_row(self.next_pos))
            self.next_row += 1
            self.next_pos += self.rng.uniform(self.cfg.row_gap_min_m, self.cfg.row_gap_max_m)
        return out

    def _make_row(self, pos: float) -> list[Obstacle]:
        names, weights = zip(*_PATTERNS)
        pattern = self.rng.choices(names, weights)[0]
        lanes = list(range(NUM_LANES))
        kinds: dict[int, Kind] = {}

        if pattern == "single":
            kinds[self.rng.choice(lanes)] = self.rng.choice(list(Kind))
        elif pattern == "two_trains":
            free = self.rng.choice(lanes)
            kinds = {l: Kind.TRAIN for l in lanes if l != free}
        elif pattern == "wall_low":
            kinds = {l: Kind.LOW for l in lanes}
        elif pattern == "wall_high":
            kinds = {l: Kind.HIGH for l in lanes}
        else:  # mixed
            order = [Kind.TRAIN, Kind.LOW, Kind.HIGH]
            self.rng.shuffle(order)
            kinds = dict(zip(lanes, order))

        # Invariant (tested): no row is all trains, so every row is passable.
        out = []
        for lane in sorted(kinds):
            kind = kinds[lane]
            length = self.cfg.train_length_m if kind is Kind.TRAIN else self.cfg.barrier_length_m
            out.append(Obstacle(self.next_id, self.next_row, lane, kind, pos, length))
            self.next_id += 1
        return out

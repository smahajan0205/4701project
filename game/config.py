"""Tunable constants for the runner. Everything here is logged with each session."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

NUM_LANES = 3


@dataclass
class GameConfig:
    seed: int = 0
    fps: int = 60

    # Run mode. "classic": the first crash ends the run (score = distance).
    # "timed": the run lasts duration_s; crashes are counted, not fatal. Timed
    # mode gives every condition identical exposure, so use it for the study.
    mode: str = "classic"
    duration_s: float = 120.0
    invulnerable_s: float = 1.0  # timed mode: grace period after a crash

    # Speed in meters per second; increases linearly up to max_speed.
    start_speed: float = 10.0
    max_speed: float = 16.0
    accel: float = 0.05

    # Action durations.
    jump_s: float = 0.65
    roll_s: float = 0.65
    lane_anim_s: float = 0.12  # visual only; the logical lane changes immediately

    # Obstacle layout (meters along the track).
    first_row_m: float = 40.0
    row_gap_min_m: float = 18.0
    row_gap_max_m: float = 30.0
    train_length_m: float = 6.0
    barrier_length_m: float = 1.0
    view_distance_m: float = 60.0

    # Events below this confidence are logged but ignored.
    min_confidence: float = 0.0

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    def frames(self, seconds: float) -> int:
        return max(1, round(seconds * self.fps))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GameConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

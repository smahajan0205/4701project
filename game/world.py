"""Deterministic game simulation. No pygame, no wall clock.

The world advances in fixed steps of 1/fps seconds. Given the same config
(seed included) and the same actions on the same frames, it reproduces a run
exactly; replay.py relies on this.

Per frame, step() does the following in order:
  1. apply this frame's actions
  2. advance distance and speed
  3. spawn obstacles that came into view
  4. check collisions at the new position
  5. retire obstacles behind the player and count down the jump/roll timers
"""

from __future__ import annotations

from common.events import EventType

from .config import NUM_LANES, GameConfig
from .obstacles import Kind, Obstacle, ObstacleGenerator


class World:
    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.gen = ObstacleGenerator(cfg)
        self.frame = 0
        self.distance = 0.0
        self.speed = cfg.start_speed
        self.lane = NUM_LANES // 2
        self.prev_lane = self.lane
        self.lane_change_frame = -10**9
        self.jump_left = 0
        self.roll_left = 0
        self.invulnerable_left = 0
        self.obstacles: list[Obstacle] = []
        # Per-obstacle outcome: "none" -> never shared its lane while overlapping,
        # "cleared" -> overlapped it but jumped/rolled correctly, "hit", "invulnerable".
        self._outcome: dict[int, str] = {}
        self.crashes = 0
        self.passed = {"none": 0, "cleared": 0, "hit": 0, "invulnerable": 0}
        self.over = False
        self.end_reason: str | None = None
        self.obstacles.extend(self._spawn())

    # ------------------------------------------------------------------ state

    @property
    def elapsed_s(self) -> float:
        return self.frame * self.cfg.dt

    @property
    def airborne(self) -> bool:
        return self.jump_left > 0

    @property
    def rolling(self) -> bool:
        return self.roll_left > 0

    @property
    def score(self) -> int:
        return int(self.distance)

    def jump_phase(self) -> float:
        """0 -> 1 over the course of a jump (for rendering)."""
        total = self.cfg.frames(self.cfg.jump_s)
        return 1.0 - self.jump_left / total if self.airborne else 0.0

    def snapshot(self) -> dict:
        return {"f": self.frame, "d": round(self.distance, 3), "v": round(self.speed, 3),
                "lane": self.lane, "air": self.jump_left, "roll": self.roll_left}

    def summary(self) -> dict:
        return {
            "frames": self.frame,
            "elapsed_s": round(self.elapsed_s, 3),
            "distance": round(self.distance, 3),
            "score": self.score,
            "crashes": self.crashes,
            "obstacles_passed": dict(self.passed),
            "end_reason": self.end_reason,
        }

    # --------------------------------------------------------------- dynamics

    def apply(self, action: EventType) -> str:
        """Apply one action. Returns its effect, for logging."""
        if self.over:
            return "ignored_over"
        if action is EventType.LEFT or action is EventType.RIGHT:
            target = self.lane + (-1 if action is EventType.LEFT else 1)
            if not 0 <= target < NUM_LANES:
                return "blocked_edge"
            self.prev_lane, self.lane = self.lane, target
            self.lane_change_frame = self.frame
            return "moved"
        if action is EventType.JUMP:
            if self.airborne:
                return "ignored_airborne"
            self.roll_left = 0
            self.jump_left = self.cfg.frames(self.cfg.jump_s)
            return "jumped"
        if action is EventType.ROLL:
            if self.rolling:
                return "ignored_rolling"
            self.jump_left = 0  # rolling mid-air drops you to the ground
            self.roll_left = self.cfg.frames(self.cfg.roll_s)
            return "rolled"
        raise ValueError(action)

    def step(self, actions: list[EventType] = ()) -> list[dict]:
        """Advance one frame. Returns world events (spawn/crash/pass/end) for logging."""
        if self.over:
            return []
        events: list[dict] = []
        cfg = self.cfg
        for a in actions:
            self.apply(a)

        self.distance += self.speed * cfg.dt
        self.speed = min(cfg.max_speed, self.speed + cfg.accel * cfg.dt)

        for o in self._spawn():
            self.obstacles.append(o)
            events.append({"type": "spawn", **o.to_dict()})

        for o in self.obstacles:
            if o.lane != self.lane or not (o.start <= self.distance <= o.end):
                continue
            if self._outcome.get(o.id, "none") in ("hit", "invulnerable"):
                continue
            if self._clears(o):
                self._outcome[o.id] = "cleared"
            elif self.invulnerable_left > 0:
                self._outcome[o.id] = "invulnerable"
            else:
                self._outcome[o.id] = "hit"
                self.crashes += 1
                events.append({"type": "crash", "obstacle": o.id, "kind": o.kind.value,
                               "lane": o.lane, "state": self.snapshot()})
                if cfg.mode == "classic":
                    self._end("crash")
                else:
                    self.invulnerable_left = cfg.frames(cfg.invulnerable_s)

        keep = []
        for o in self.obstacles:
            if o.end < self.distance:
                outcome = self._outcome.pop(o.id, "none")
                self.passed[outcome] += 1
                events.append({"type": "pass", "obstacle": o.id, "row": o.row,
                               "kind": o.kind.value, "lane": o.lane, "outcome": outcome})
            else:
                keep.append(o)
        self.obstacles = keep

        self.jump_left = max(0, self.jump_left - 1)
        self.roll_left = max(0, self.roll_left - 1)
        self.invulnerable_left = max(0, self.invulnerable_left - 1)
        self.frame += 1

        if cfg.mode == "timed" and not self.over and self.elapsed_s >= cfg.duration_s:
            self._end("time")
        if self.over:
            events.append({"type": "end", **self.summary()})
        return events

    # -------------------------------------------------------------- internals

    def _clears(self, o: Obstacle) -> bool:
        if o.kind is Kind.LOW:
            return self.airborne
        if o.kind is Kind.HIGH:
            return self.rolling
        return False

    def _spawn(self) -> list[Obstacle]:
        return self.gen.generate_until(self.distance + self.cfg.view_distance_m)

    def _end(self, reason: str) -> None:
        self.over = True
        self.end_reason = reason

"""A simulated player + classifier, for testing and for latency-budget analysis.

The simulated player sees each obstacle row once it is ``react_s`` seconds
away and decides what to do (change lanes, jump, roll). Every intended action
then goes through a simulated classifier that:
  * delivers it ``latency_ms`` later,
  * swaps it for a random wrong action with probability ``1 - accuracy``,
  * also fires spurious actions at ``fa_per_min`` (false activations).

Sweeping these knobs (see sweep.py) estimates how classifier operating points
(latency, accuracy, false activations) turn into game outcomes. The point is to
find the latency budget the game tolerates before real users ever play.

The player is deliberately naive: it does not compensate for latency. It times
jumps and rolls to be centered on the obstacle as if input were instantaneous.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import asdict, dataclass

from common.events import EventType

from .config import NUM_LANES
from .obstacles import Kind
from .world import World

_ALL = list(EventType)


def _lane_delta(action: EventType) -> int:
    return {EventType.LEFT: -1, EventType.RIGHT: 1}.get(action, 0)


@dataclass
class AutopilotConfig:
    latency_ms: float = 0.0
    accuracy: float = 1.0
    fa_per_min: float = 0.0
    react_s: float = 1.0
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class Autopilot:
    def __init__(self, world: World, cfg: AutopilotConfig):
        self.w = world
        self.cfg = cfg
        self.rng = random.Random(cfg.seed * 1_000_003 + 7)
        self.latency_frames = round(cfg.latency_ms / 1000 * world.cfg.fps)
        self._planned_rows: set[int] = set()
        self._last_row_end = float("-inf")
        self._queue: list[tuple[int, int, int, EventType, EventType]] = []  # heap
        self._seq = 0
        self._next_fa = self._draw_fa_frame(0)
        self._lane_offset = 0  # lane moves emitted/queued but not yet applied

    def poll(self) -> list[tuple[EventType, EventType | None, int]]:
        """Actions due on the current frame, as (emitted, intended, intended_frame).

        `intended` is None for a false activation; when it differs from `emitted`,
        the event was a misclassification.
        """
        f = self.w.frame
        self._plan(f)
        out = []
        while self._queue and self._queue[0][0] <= f:
            _, _, intended_f, emitted, intended = heapq.heappop(self._queue)
            self._lane_offset -= _lane_delta(emitted)
            out.append((emitted, intended, intended_f))
        if self.cfg.fa_per_min > 0 and f >= self._next_fa:
            out.append((self.rng.choice(_ALL), None, f))
            self._next_fa = self._draw_fa_frame(f)
        return out

    # -------------------------------------------------------------- internals

    def _draw_fa_frame(self, f: int) -> int:
        if self.cfg.fa_per_min <= 0:
            return 1 << 62
        gap_s = self.rng.expovariate(self.cfg.fa_per_min / 60.0)
        return f + max(1, round(gap_s * self.w.cfg.fps))

    def _emit(self, action: EventType, intended_frame: int) -> None:
        emitted = action
        if self.rng.random() > self.cfg.accuracy:
            emitted = self.rng.choice([a for a in _ALL if a is not action])
        self._lane_offset += _lane_delta(emitted)
        release = intended_frame + self.latency_frames
        heapq.heappush(self._queue, (release, self._seq, intended_frame, emitted, action))
        self._seq += 1

    def _plan(self, f: int) -> None:
        w = self.w
        rows: dict[int, list] = {}
        for o in w.obstacles:
            if o.row not in self._planned_rows:
                rows.setdefault(o.row, []).append(o)
        for row in sorted(rows):
            obs = rows[row]
            start = min(o.start for o in obs)
            # Respond to one row at a time: wait until the previous one is behind us.
            if w.distance <= self._last_row_end:
                return
            arrival_s = (start - w.distance) / w.speed
            if arrival_s > self.cfg.react_s:
                return
            self._plan_row(f, obs, start)
            self._planned_rows.add(row)
            self._last_row_end = max(o.end for o in obs)

    def _plan_row(self, f: int, obs: list, start: float) -> None:
        w = self.w
        kinds = {o.lane: o.kind for o in obs}
        here = min(max(w.lane + self._lane_offset, 0), NUM_LANES - 1)

        def cost(lane: int) -> tuple:
            k = kinds.get(lane)
            return (k is Kind.TRAIN, abs(lane - here), k is not None, self.rng.random())

        target = min(range(NUM_LANES), key=cost)
        step = EventType.LEFT if target < here else EventType.RIGHT
        for i in range(abs(target - here)):
            self._emit(step, f + i)

        kind = kinds.get(target)
        if kind in (Kind.LOW, Kind.HIGH):
            action = EventType.JUMP if kind is Kind.LOW else EventType.ROLL
            dur = w.cfg.frames(w.cfg.jump_s if kind is Kind.LOW else w.cfg.roll_s)
            arrival_f = f + round((start - w.distance) / w.speed * w.cfg.fps)
            self._emit(action, max(f + abs(target - here), arrival_f - dur // 2))

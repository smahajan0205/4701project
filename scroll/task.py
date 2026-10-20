"""Scrolling target-acquisition task. No pygame, no wall clock.

The document is a column of equal-height items grouped into sections. Actions
map onto the same four gesture events as the runner:

  roll  (swipe down)  scroll down half a viewport
  jump  (swipe up)    scroll up half a viewport
  right               jump to the next section start
  left                jump to the previous section start

A trial highlights one target item; it is acquired as soon as the whole item is
on screen. For every target we also compute, by breadth-first search over
reachable offsets, the fewest actions that would have acquired it, so
"excess inputs" measures navigation cost independently of target distance.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass

from common.events import EventType


@dataclass
class ScrollConfig:
    seed: int = 0
    n_items: int = 200
    item_h: int = 120
    view_h: int = 600
    section_items: int = 10
    n_targets: int = 15
    min_jump: int = 5    # items between consecutive targets
    max_jump: int = 40

    @property
    def step(self) -> int:
        return self.view_h // 2

    @property
    def section_h(self) -> int:
        return self.section_items * self.item_h

    @property
    def max_offset(self) -> int:
        return self.n_items * self.item_h - self.view_h

    def to_dict(self) -> dict:
        return asdict(self)


def next_offset(cfg: ScrollConfig, offset: int, action: EventType) -> int:
    if action is EventType.ROLL:
        new = offset + cfg.step
    elif action is EventType.JUMP:
        new = offset - cfg.step
    elif action is EventType.RIGHT:
        new = (offset // cfg.section_h + 1) * cfg.section_h
    else:  # LEFT: start of this section, or of the previous one if already there
        new = (offset - 1) // cfg.section_h * cfg.section_h
    return min(max(new, 0), cfg.max_offset)


def visible(cfg: ScrollConfig, offset: int, item: int) -> bool:
    top = item * cfg.item_h - offset
    return top >= 0 and top + cfg.item_h <= cfg.view_h


def optimal_inputs(cfg: ScrollConfig, offset: int, item: int) -> int:
    seen, frontier = {offset}, deque([(offset, 0)])
    while frontier:
        off, n = frontier.popleft()
        if visible(cfg, off, item):
            return n
        for a in EventType:
            nxt = next_offset(cfg, off, a)
            if nxt not in seen:
                seen.add(nxt)
                frontier.append((nxt, n + 1))
    raise ValueError(f"item {item} unreachable")


def make_targets(cfg: ScrollConfig) -> list[int]:
    rng = random.Random(cfg.seed)
    out, cur = [], 0
    while len(out) < cfg.n_targets:
        jump = rng.randint(cfg.min_jump, cfg.max_jump) * rng.choice((-1, 1))
        if 0 <= cur + jump < cfg.n_items:
            cur += jump
            out.append(cur)
    return out


class ScrollTask:
    def __init__(self, cfg: ScrollConfig):
        self.cfg = cfg
        self.targets = make_targets(cfg)
        self.offset = 0
        self.index = 0
        self.done = False
        self.results: list[dict] = []
        self._begin(0.0)

    @property
    def target(self) -> int:
        return self.targets[self.index]

    def _begin(self, t: float) -> None:
        self._t0 = t
        self._inputs = 0
        self._reversals = 0
        self._last_dir = 0
        self.optimal = optimal_inputs(self.cfg, self.offset, self.target)

    def apply(self, action: EventType, t: float) -> list[dict]:
        """Apply one action at task time t (s). Returns task events for logging."""
        if self.done:
            return []
        before = self.offset
        self.offset = next_offset(self.cfg, before, action)
        self._inputs += 1
        direction = (self.offset > before) - (self.offset < before)
        if direction and self._last_dir and direction != self._last_dir:
            self._reversals += 1
        self._last_dir = direction or self._last_dir
        return self.check(t)

    def check(self, t: float) -> list[dict]:
        if self.done or not visible(self.cfg, self.offset, self.target):
            return []
        r = {"type": "acquired", "trial": self.index, "target": self.target,
             "time_s": round(t - self._t0, 4), "inputs": self._inputs, "optimal": self.optimal,
             "excess": self._inputs - self.optimal, "reversals": self._reversals}
        self.results.append(r)
        events = [r]
        self.index += 1
        if self.index == len(self.targets):
            self.done = True
        else:
            self._begin(t)
            events.append({"type": "target", "trial": self.index, "target": self.target,
                           "offset": self.offset, "optimal": self.optimal})
            events += self.check(t)  # already on screen: acquired with zero inputs
        return events

    def summary(self) -> dict:
        rs = self.results
        n = len(rs) or 1
        return {"acquired": len(rs), "of": len(self.targets),
                "mean_time_s": round(sum(r["time_s"] for r in rs) / n, 3),
                "total_inputs": sum(r["inputs"] for r in rs),
                "total_excess": sum(r["excess"] for r in rs),
                "total_reversals": sum(r["reversals"] for r in rs)}

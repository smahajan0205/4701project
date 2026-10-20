import threading

import pytest

from common.events import EventReceiver, EventSender, EventType, GestureEvent, now
from game.autopilot import Autopilot, AutopilotConfig
from game.config import GameConfig
from game.obstacles import Kind, Obstacle, ObstacleGenerator
from game.world import World


def run_autopilot(seed=0, duration=30.0, **ap):
    w = World(GameConfig(seed=seed, mode="timed", duration_s=duration))
    pilot = Autopilot(w, AutopilotConfig(seed=seed, **ap))
    while not w.over:
        for a, _, _ in pilot.poll():
            w.apply(a)
        w.step()
    return w


def place(w: World, kind: Kind, lane: int, ahead: float = 0.2) -> Obstacle:
    """Clear the track and put one obstacle just ahead of the player."""
    length = w.cfg.train_length_m if kind is Kind.TRAIN else w.cfg.barrier_length_m
    o = Obstacle(10_000, 10_000, lane, kind, w.distance + ahead, length)
    w.obstacles = [o]
    w.gen.next_pos = float("inf")
    return o


# ----------------------------------------------------------------- generator

def test_same_seed_same_course():
    a, b = ObstacleGenerator(GameConfig(seed=7)), ObstacleGenerator(GameConfig(seed=7))
    assert [o.to_dict() for o in a.generate_until(2000)] == [o.to_dict() for o in b.generate_until(2000)]


def test_different_seed_different_course():
    a, b = ObstacleGenerator(GameConfig(seed=1)), ObstacleGenerator(GameConfig(seed=2))
    assert [o.to_dict() for o in a.generate_until(500)] != [o.to_dict() for o in b.generate_until(500)]


@pytest.mark.parametrize("seed", range(20))
def test_every_row_is_passable(seed):
    rows = {}
    for o in ObstacleGenerator(GameConfig(seed=seed)).generate_until(5000):
        rows.setdefault(o.row, []).append(o)
    for obs in rows.values():
        assert sum(o.kind is Kind.TRAIN for o in obs) < 3


# --------------------------------------------------------------------- rules

def test_lane_changes_and_edges():
    w = World(GameConfig())
    assert w.lane == 1
    assert w.apply(EventType.LEFT) == "moved" and w.lane == 0
    assert w.apply(EventType.LEFT) == "blocked_edge" and w.lane == 0
    w.apply(EventType.RIGHT)
    w.apply(EventType.RIGHT)
    assert w.apply(EventType.RIGHT) == "blocked_edge" and w.lane == 2


@pytest.mark.parametrize("kind,action,survives", [
    (Kind.LOW, EventType.JUMP, True), (Kind.LOW, EventType.ROLL, False), (Kind.LOW, None, False),
    (Kind.HIGH, EventType.ROLL, True), (Kind.HIGH, EventType.JUMP, False), (Kind.HIGH, None, False),
    (Kind.TRAIN, EventType.JUMP, False), (Kind.TRAIN, EventType.ROLL, False),
])
def test_obstacle_rules(kind, action, survives):
    w = World(GameConfig(mode="classic"))
    place(w, kind, w.lane)
    if action:
        w.apply(action)
    for _ in range(60):
        w.step()
    assert (w.end_reason != "crash") == survives


def test_dodging_a_train_by_changing_lanes():
    w = World(GameConfig(mode="classic"))
    place(w, Kind.TRAIN, w.lane)
    w.apply(EventType.LEFT)
    for _ in range(120):
        w.step()
    assert not w.over and w.passed["none"] == 1


def test_switching_into_a_train_alongside_crashes():
    w = World(GameConfig(mode="classic"))
    place(w, Kind.TRAIN, 0, ahead=-1.0)  # train already beside us in lane 0
    w.step()
    w.apply(EventType.LEFT)
    w.step()
    assert w.end_reason == "crash"


def test_timed_mode_counts_crashes_and_ends_on_time():
    w = World(GameConfig(mode="timed", duration_s=5))
    place(w, Kind.TRAIN, w.lane)
    while not w.over:
        w.step()
    assert w.crashes == 1 and w.end_reason == "time"
    assert w.frame == 5 * w.cfg.fps


# ----------------------------------------------------------------- autopilot

@pytest.mark.parametrize("seed", range(5))
def test_autopilot_with_no_latency_never_crashes(seed):
    assert run_autopilot(seed, duration=60).crashes == 0


def test_large_latency_causes_crashes():
    assert run_autopilot(0, duration=60, latency_ms=600).crashes > 5


def test_simulation_is_deterministic():
    a = run_autopilot(3, latency_ms=200, accuracy=0.9, fa_per_min=4)
    b = run_autopilot(3, latency_ms=200, accuracy=0.9, fa_per_min=4)
    assert a.summary() == b.summary()


# -------------------------------------------------------------------- events

def test_event_json_roundtrip_and_aliases():
    e = GestureEvent(EventType.JUMP, 0.8, 12.5, t_capture=12.4, stages={"t_inference": 12.45}, source="x")
    assert GestureEvent.from_json(e.to_json()) == e
    assert EventType.parse("up") is EventType.JUMP
    assert EventType.parse("DOWN") is EventType.ROLL
    with pytest.raises(ValueError):
        EventType.parse("idle")


def test_udp_send_and_receive():
    rx = EventReceiver(port=0)
    port = rx.sock.getsockname()[1]
    tx = EventSender(port=port, source="test")
    sent = tx.send("left", 0.9, t_capture=now())
    got = []
    deadline = now() + 1.0
    while not got and now() < deadline:
        got = rx.poll()
        threading.Event().wait(0.005)
    tx.close()
    rx.close()
    assert len(got) == 1 and got[0][0] == sent and got[0][1] >= sent.timestamp

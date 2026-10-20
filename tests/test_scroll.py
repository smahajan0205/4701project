import pytest

from common.events import EventType
from scroll.task import ScrollConfig, ScrollTask, make_targets, next_offset, optimal_inputs, visible

CFG = ScrollConfig()


def greedy(task: ScrollTask) -> EventType:
    """Scroll toward the target a half-page at a time."""
    top = task.target * task.cfg.item_h - task.offset
    return EventType.ROLL if top > 0 else EventType.JUMP


def test_moves_and_clamps():
    assert next_offset(CFG, 0, EventType.ROLL) == CFG.step
    assert next_offset(CFG, 0, EventType.JUMP) == 0
    assert next_offset(CFG, 0, EventType.RIGHT) == CFG.section_h
    assert next_offset(CFG, CFG.section_h, EventType.LEFT) == 0
    assert next_offset(CFG, CFG.section_h + 1, EventType.LEFT) == CFG.section_h
    assert next_offset(CFG, CFG.max_offset, EventType.ROLL) == CFG.max_offset


def test_optimal_inputs_small_cases():
    assert optimal_inputs(CFG, 0, 0) == 0
    assert optimal_inputs(CFG, 0, 6) == 1     # one half-page down
    assert optimal_inputs(CFG, 0, 10) == 1    # next section
    assert optimal_inputs(CFG, 0, 30) == 3    # three sections


def test_every_item_is_reachable_from_anywhere():
    for item in range(0, CFG.n_items, 7):
        assert optimal_inputs(CFG, 0, item) >= 0
        assert optimal_inputs(CFG, CFG.max_offset, item) >= 0


def test_targets_are_seeded_and_in_range():
    assert make_targets(ScrollConfig(seed=3)) == make_targets(ScrollConfig(seed=3))
    assert make_targets(ScrollConfig(seed=3)) != make_targets(ScrollConfig(seed=4))
    assert all(0 <= t < CFG.n_items for t in make_targets(CFG))


@pytest.mark.parametrize("seed", range(5))
def test_greedy_player_finishes_with_nonnegative_excess(seed):
    task = ScrollTask(ScrollConfig(seed=seed))
    t = 0.0
    while not task.done:
        t += 0.5
        task.apply(greedy(task), t)
        assert t < 1000
    s = task.summary()
    assert s["acquired"] == s["of"] == CFG.n_targets
    assert all(r["excess"] >= 0 for r in task.results)


def test_overshoot_counts_a_reversal():
    task = ScrollTask(ScrollConfig(seed=0, n_targets=1))
    task.targets = [8]
    task._begin(0.0)
    assert task.apply(EventType.RIGHT, 1.0) == []    # overshoots to item 10
    assert not visible(task.cfg, task.offset, 8)
    [r] = task.apply(EventType.JUMP, 2.0)              # back up half a page
    assert task.done
    assert r == {"type": "acquired", "trial": 0, "target": 8, "time_s": 2.0, "inputs": 2,
                 "optimal": 2, "excess": 0, "reversals": 1}

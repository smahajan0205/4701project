import json
from collections import Counter, deque

import pytest

from common.events import EventType
from game.app import AppOptions, run
from game.autopilot import AutopilotConfig
from game.config import GameConfig
from game.logger import read_log
from integration.threshold_bridge import detect
from study import protocol
from study.run import run_session
from study.summary import rows

# ------------------------------------------------------------------ protocol


def test_first_six_participants_are_fully_counterbalanced():
    orders = [protocol.order_for(f"P{n:02d}") for n in range(1, 7)]
    assert len(set(orders)) == 6
    for pos in range(3):
        assert Counter(o[pos] for o in orders) == {c: 2 for c in protocol.CONDITIONS}
    pairs = Counter((o[i], o[i + 1]) for o in orders for i in range(2))
    assert len(pairs) == 6 and set(pairs.values()) == {2}


def test_seed_depends_only_on_session():
    a, b = protocol.schedule("P01", 2), protocol.schedule("P05", 2)
    assert {t.seed for t in a} == {t.seed for t in b} == {protocol.session_seed(2)}
    assert protocol.session_seed(1) != protocol.session_seed(2)


def test_participant_id_must_end_in_number():
    with pytest.raises(ValueError):
        protocol.order_for("alice")


@pytest.mark.parametrize("raw,want", [("0", 0), ("100", 100), ("42", 40), ("43", 45), ("7.5", 10)])
def test_tlx_rounds_to_scale(raw, want):
    assert protocol.parse_tlx(raw) == want


def test_tlx_reprompts_on_bad_input():
    answers = iter(["abc", "150", "50", "10", "20", "30", "40", "50"])
    out = protocol.administer_tlx(ask=lambda _: next(answers), say=lambda *_: None)
    assert out["mental"] == 50 and out["physical"] == 10 and out["frustration"] == 50
    assert out["raw_tlx"] == 33.33


# ------------------------------------------------------------------- session


def autopilot_play(trial, duration_s, participant, label, log_dir, seed=None):
    """Stand-in for study.run.play: the real game loop, headless, autopilot input."""
    opts = AppOptions(game=GameConfig(seed=trial.seed if seed is None else seed, mode="timed",
                                      duration_s=duration_s, fps=240),
                      input="autopilot", autopilot=AutopilotConfig(latency_ms=200, seed=1),
                      participant=participant, condition=label, log_dir=log_dir,
                      countdown_s=0, headless=True, exit_on_end=True)
    [path] = run(opts)
    end = read_log(path)[-1]
    return {"log": str(path), **{k: end[k] for k in ("end_reason", "score", "crashes")}}


def test_full_session_writes_record_and_summary(tmp_path):
    ask = lambda prompt: "y" if "consent" in prompt else "50"  # noqa: E731
    path = run_session("P02", 1, study_dir=tmp_path, practice_s=0, trial_s=2,
                       ask=ask, say=lambda *_: None, play_fn=autopilot_play)
    rec = json.loads(path.read_text())
    assert "consent" in rec and "finished" in rec
    assert [t["condition"] for t in rec["trials"]] == list(protocol.order_for("P02"))
    assert all(t["tlx"]["raw_tlx"] == 50 for t in rec["trials"])
    labels = [read_log(t["log"])[0]["condition"] for t in rec["trials"]]
    assert labels == rec["order"]
    assert len(rows(tmp_path)) == 3


def test_no_consent_stops_before_playing(tmp_path):
    with pytest.raises(SystemExit):
        run_session("P01", 1, study_dir=tmp_path, ask=lambda _: "n", say=lambda *_: None,
                    play_fn=pytest.fail)


def test_resume_skips_completed_conditions(tmp_path):
    calls = []

    def flaky(trial, duration_s, participant, label, log_dir, seed=None):
        calls.append(label)
        quit_early = len(calls) == 2
        return {"log": None, "end_reason": "quit" if quit_early else "time", "score": 1, "crashes": 0}

    quiet = dict(study_dir=tmp_path, practice_s=0, trial_s=1, ask=lambda _: "50",
                 say=lambda *_: None, play_fn=flaky)
    with pytest.raises(SystemExit):
        run_session("P03", 2, **quiet)
    run_session("P03", 2, resume=True, **quiet)
    order = list(protocol.order_for("P03"))
    assert calls == [order[0], order[1], order[1], order[2]]
    rec = json.loads((tmp_path / "P03" / "session2.json").read_text())
    assert [t["end_reason"] for t in rec["trials"]] == ["time", "quit", "time", "time"]
    assert len(rows(tmp_path)) == 3


# --------------------------------------------------------- velocity baseline


def swipe(dx, dy, n=6, hand_len=0.1):
    return deque((i / 30, 0.5 + dx * i / (n - 1), 0.5 + dy * i / (n - 1), hand_len) for i in range(n))


@pytest.mark.parametrize("dx,dy,want", [(0.2, 0, EventType.RIGHT), (-0.2, 0, EventType.LEFT),
                                        (0, -0.2, EventType.JUMP), (0, 0.2, EventType.ROLL)])
def test_threshold_detects_each_direction(dx, dy, want):
    kind, conf = detect(swipe(dx, dy), thresh=1.2)
    assert kind is want and 0 < conf <= 1


def test_threshold_ignores_small_and_diagonal_motion():
    assert detect(swipe(0.05, 0), thresh=1.2) is None      # 0.5 hand-lengths
    assert detect(swipe(0.2, 0.2), thresh=1.2) is None     # ambiguous diagonal
    assert detect(swipe(0.2, 0, n=2), thresh=1.2) is None  # too few frames

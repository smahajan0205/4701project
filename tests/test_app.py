"""End-to-end: run the real pygame loop headlessly, then replay its log."""

from game.app import AppOptions, run
from game.autopilot import AutopilotConfig
from game.config import GameConfig
from game.latency_report import segments
from game.logger import read_log
from game.replay import verify


def test_headless_session_logs_and_replays(tmp_path):
    opts = AppOptions(
        game=GameConfig(seed=4, mode="timed", duration_s=6, fps=240),
        input="autopilot", autopilot=AutopilotConfig(latency_ms=150, accuracy=0.8, fa_per_min=20, seed=4),
        countdown_s=0, log_dir=tmp_path, headless=True, exit_on_end=True, participant="test",
    )
    [path] = run(opts)
    records = read_log(path)
    types = {r["type"] for r in records}
    assert {"session_start", "input", "frame", "spawn", "session_end"} <= types
    end = records[-1]
    assert end["type"] == "session_end" and end["end_reason"] == "time"

    inputs = [r for r in records if r["type"] == "input" and r["accepted"]]
    assert inputs
    for r in inputs:
        names = [name for name, _ in segments(r)]
        assert names[0] == "capture->emit" and names[-1].startswith("TOTAL")
        assert r["t_recv"] <= r["t_applied"] <= r["t_displayed"]

    assert verify(path)

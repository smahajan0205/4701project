"""Play the runner.

    python -m game                               # keyboard, seed 0, classic mode
    python -m game --input gesture               # listen for classifier events on UDP 5005
    python -m game --input both                  # gesture + keyboard (for development)
    python -m game --mode timed --duration 120 --seed 3 --participant P01 --condition gesture
    python -m game --input autopilot --autopilot-latency-ms 250

Keys: arrows / WASD / space to play, P pause, R restart after game over, Esc quit.
"""

import argparse
from pathlib import Path

from common.events import DEFAULT_PORT

from .app import INPUT_MODES, AppOptions, run
from .autopilot import AutopilotConfig
from .config import GameConfig
from .logger import LOG_DIR


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m game", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", choices=INPUT_MODES, default="keyboard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=("classic", "timed"), default="classic")
    p.add_argument("--duration", type=float, default=120.0, help="seconds (timed mode)")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--participant", default="dev")
    p.add_argument("--condition", default="")
    p.add_argument("--countdown", type=float, default=3.0, help="seconds before the run starts")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--no-frame-log", action="store_true")
    p.add_argument("--log-dir", type=Path, default=LOG_DIR)
    p.add_argument("--headless", action="store_true", help="no window (SDL dummy driver)")
    p.add_argument("--exit-on-end", action="store_true")
    p.add_argument("--autopilot-latency-ms", type=float, default=0.0)
    p.add_argument("--autopilot-accuracy", type=float, default=1.0)
    p.add_argument("--autopilot-fa-per-min", type=float, default=0.0)
    a = p.parse_args(argv)

    opts = AppOptions(
        game=GameConfig(seed=a.seed, fps=a.fps, mode=a.mode, duration_s=a.duration,
                        min_confidence=a.min_confidence),
        input=a.input, port=a.port, participant=a.participant, condition=a.condition,
        autopilot=AutopilotConfig(latency_ms=a.autopilot_latency_ms, accuracy=a.autopilot_accuracy,
                                  fa_per_min=a.autopilot_fa_per_min, seed=a.seed),
        countdown_s=a.countdown, log=not a.no_log, log_frames=not a.no_frame_log,
        log_dir=a.log_dir, headless=a.headless, exit_on_end=a.exit_on_end,
    )
    for path in run(opts):
        print(f"log: {path}")


if __name__ == "__main__":
    main()

"""Latency-budget sweep: how do classifier latency, accuracy, and false
activations translate into game outcomes? Runs the autopilot headlessly
(no rendering, faster than real time) over a grid of operating points.

    python -m game.sweep
    python -m game.sweep --latencies 0,100,200,300,400,500 --accuracies 1,0.9,0.8 --seeds 10
    python -m game.sweep --fa-per-min 0,2,5 --csv results/sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics

from .autopilot import Autopilot, AutopilotConfig
from .config import GameConfig
from .world import World


def run_one(seed: int, latency_ms: float, accuracy: float, fa_per_min: float,
            duration_s: float, react_s: float) -> dict:
    world = World(GameConfig(seed=seed, mode="timed", duration_s=duration_s))
    ap = Autopilot(world, AutopilotConfig(latency_ms=latency_ms, accuracy=accuracy,
                                          fa_per_min=fa_per_min, react_s=react_s, seed=seed))
    while not world.over:
        for emitted, _, _ in ap.poll():
            world.apply(emitted)
        world.step()
    p = world.passed
    engaged = p["cleared"] + p["hit"]
    return {
        "seed": seed, "latency_ms": latency_ms, "accuracy": accuracy, "fa_per_min": fa_per_min,
        "crashes": world.crashes,
        "crashes_per_min": world.crashes / (duration_s / 60),
        "distance": round(world.distance, 1),
        "obstacles": sum(p.values()),
        "engaged_clear_rate": p["cleared"] / engaged if engaged else 1.0,
    }


def floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m game.sweep", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latencies", type=floats, default=floats("0,100,200,250,300,350,400,500,700"))
    p.add_argument("--accuracies", type=floats, default=[1.0])
    p.add_argument("--fa-per-min", type=floats, default=[0.0])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--react", type=float, default=1.0, help="simulated reaction lead time (s)")
    p.add_argument("--csv")
    a = p.parse_args(argv)

    rows = []
    print(f"{'latency':>8s} {'acc':>5s} {'FA/min':>6s}   crashes/min (mean ± sd over {a.seeds} seeds)")
    for lat, acc, fa in itertools.product(a.latencies, a.accuracies, a.fa_per_min):
        runs = [run_one(s, lat, acc, fa, a.duration, a.react) for s in range(a.seeds)]
        rows += runs
        cpm = [r["crashes_per_min"] for r in runs]
        sd = statistics.stdev(cpm) if len(cpm) > 1 else 0.0
        print(f"{lat:8.0f} {acc:5.2f} {fa:6.1f}   {statistics.fmean(cpm):6.2f} ± {sd:.2f}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()

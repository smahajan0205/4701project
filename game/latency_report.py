"""Decompose input latency from one or more session logs.

    python -m game.latency_report logs/*.jsonl
    python -m game.latency_report logs/*.jsonl --csv latency.csv

For every accepted input it splits the time from camera capture to the frame
that shows the result into segments:

  t_capture -> [stage timestamps the classifier reported, in time order] -> emit
  emit      -> recv        UDP transport + waiting for the game's next poll (the
                           socket is drained once per frame, so up to 1/fps)
  recv      -> applied     input handling within the frame (near zero)
  applied   -> displayed   rendering + display.flip()

Not covered: camera sensor exposure/readout before cap.read() returns, and
display scan-out after flip. Those need an external measurement (for example a
high-speed phone video of a hand and the screen together; see docs/game.md).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from .logger import read_log


def segments(rec: dict) -> list[tuple[str, float]]:
    ev = rec["event"]
    chain = []
    if ev.get("t_capture") is not None:
        chain.append(("capture", ev["t_capture"]))
    chain += sorted(((k.removeprefix("t_"), v) for k, v in (ev.get("stages") or {}).items()),
                    key=lambda kv: kv[1])
    chain += [("emit", ev["timestamp"]), ("recv", rec["t_recv"]),
              ("applied", rec["t_applied"]), ("displayed", rec["t_displayed"])]
    out = [(f"{a}->{b}", (tb - ta) * 1000) for (a, ta), (b, tb) in zip(chain, chain[1:])]
    out.append((f"TOTAL {chain[0][0]}->displayed", (chain[-1][1] - chain[0][1]) * 1000))
    return out


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m game.latency_report", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+")
    p.add_argument("--csv", help="write one row per (input, segment)")
    a = p.parse_args(argv)

    by_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    order: dict[str, list[str]] = defaultdict(list)
    rows = []
    frame_stats = []
    for path in a.logs:
        for r in read_log(path):
            if r["type"] == "session_end" and r.get("frame_ms"):
                frame_stats.append((path, r["frame_ms"]))
            if r["type"] != "input" or not r.get("accepted"):
                continue
            src = r["event"].get("source") or "unknown"
            for name, ms in segments(r):
                if name not in order[src]:
                    order[src].append(name)
                by_source[src][name].append(ms)
                rows.append({"log": path, "f": r["f"], "source": src, "segment": name, "ms": ms})

    if not rows:
        print("no accepted inputs found")
    for src, segs in by_source.items():
        print(f"\nsource: {src}")
        print(f"  {'segment':34s} {'n':>5s} {'median':>8s} {'p90':>8s} {'mean':>8s}   (ms)")
        for name in order[src]:
            xs = segs[name]
            print(f"  {name:34s} {len(xs):5d} {pct(xs, .5):8.1f} {pct(xs, .9):8.1f} "
                  f"{sum(xs) / len(xs):8.1f}")
    if frame_stats:
        print("\nframe timing (ms between displayed frames):")
        for path, fs in frame_stats:
            print(f"  {path}: p50 {fs['p50']}  p95 {fs['p95']}  max {fs['max']}  "
                  f"late {fs['late_frames']}/{fs['n']}")
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()

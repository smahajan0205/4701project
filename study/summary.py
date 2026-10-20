"""Collect every completed trial into one table.

    python -m study.summary                          # means per condition x session
    python -m study.summary --csv results/study.csv  # one row per trial, for analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .protocol import CONDITIONS, TLX_SCALES
from .run import STUDY_DIR


def rows(study_dir: Path) -> list[dict]:
    out = []
    for path in sorted(study_dir.glob("*/session*.json")):
        rec = json.loads(path.read_text())
        for t in rec["trials"]:
            if t.get("end_reason") != "time" or "tlx" not in t:
                continue
            out.append({"participant": rec["participant"], "session": rec["session"],
                        "position": t["position"], "condition": t["condition"], "seed": t["seed"],
                        "score": t["score"], "crashes": t["crashes"],
                        **{k: t["tlx"][k] for k, *_ in TLX_SCALES}, "raw_tlx": t["tlx"]["raw_tlx"]})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m study.summary", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study-dir", type=Path, default=STUDY_DIR)
    p.add_argument("--csv")
    a = p.parse_args(argv)

    data = rows(a.study_dir)
    if not data:
        raise SystemExit(f"no completed trials under {a.study_dir}")
    groups = defaultdict(list)
    for r in data:
        groups[(r["condition"], r["session"])].append(r)
    print(f"{len(data)} trials from {len({r['participant'] for r in data})} participants\n")
    print(f"{'condition':20s} {'session':>7s} {'n':>3s} {'crashes':>8s} {'score':>7s} {'raw TLX':>8s}")
    for cond in CONDITIONS:
        for session in sorted({s for c, s in groups if c == cond}):
            g = groups[(cond, session)]
            mean = lambda k: statistics.fmean(r[k] for r in g)  # noqa: E731
            print(f"{cond:20s} {session:7d} {len(g):3d} {mean('crashes'):8.2f} "
                  f"{mean('score'):7.0f} {mean('raw_tlx'):8.1f}")
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()

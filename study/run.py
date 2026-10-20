"""Run one participant session: every condition in counterbalanced order, a
short practice before each, the timed trial, then NASA-TLX.

    python -m study.run --participant P01 --session 1
    python -m study.run --participant P01 --session 1 --resume   # after a crash or Esc
    python -m study.run --participant P01 --session 1 --show     # print the plan only

For gesture conditions, start the classifier in another terminal first (it
sends to UDP 5005, same as for the plain game). Results go to
logs/study/<participant>/session<k>.json, written after every condition, and
the game logs for each run sit next to it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from game.app import AppOptions, run
from game.config import GameConfig
from game.logger import LOG_DIR, git_commit, read_log

from . import protocol
from .protocol import CONDITION_INFO, Trial

STUDY_DIR = LOG_DIR / "study"


def play(trial: Trial, duration_s: float, participant: str, label: str, log_dir: Path,
         seed: int | None = None) -> dict:
    """One timed run in a window. Returns the session_end summary plus the log path."""
    cfg = GameConfig(seed=trial.seed if seed is None else seed, mode="timed", duration_s=duration_s)
    opts = AppOptions(game=cfg, input=trial.input_mode, participant=participant, condition=label,
                      log_dir=log_dir, exit_on_end=True)
    paths = run(opts)
    end = read_log(paths[-1])[-1] if paths else {}
    return {"log": str(paths[-1]) if paths else None,
            **{k: end.get(k) for k in ("end_reason", "score", "distance", "crashes", "obstacles_passed")}}


def run_session(participant: str, session: int, *, study_dir: Path = STUDY_DIR,
                practice_s: float = protocol.PRACTICE_S, trial_s: float = protocol.TRIAL_S,
                resume: bool = False, ask=input, say=print, play_fn=play) -> Path:
    out_dir = study_dir / participant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"session{session}.json"
    trials = protocol.schedule(participant, session)

    if out_path.exists() and not resume:
        raise SystemExit(f"{out_path} exists; pass --resume to continue it")
    if out_path.exists():
        record = json.loads(out_path.read_text())
    else:
        record = {"participant": participant, "session": session,
                  "order": [t.condition for t in trials], "seed": trials[0].seed,
                  "trial_s": trial_s, "practice_s": practice_s, "git": git_commit(),
                  "started": dt.datetime.now().isoformat(timespec="seconds"), "trials": []}
        if session == 1:
            if ask("Has the participant signed the consent form? [y/N] ").strip().lower() != "y":
                raise SystemExit("no consent recorded; stopping")
            record["consent"] = dt.datetime.now().isoformat(timespec="seconds")

    def save():
        out_path.write_text(json.dumps(record, indent=2))

    save()
    done = {t["condition"] for t in record["trials"] if t.get("end_reason") == "time"}
    for trial in trials:
        if trial.condition in done:
            continue
        instructions = CONDITION_INFO[trial.condition][1]
        say(f"\n=== {participant} session {session}, condition {trial.position}/{len(trials)}: "
            f"{trial.condition} ===\n{instructions}")
        if trial.input_mode == "gesture":
            say("Make sure the classifier is running and the participant's hand is in view.")
        if practice_s > 0:
            ask(f"Press Enter to start a {practice_s:.0f} s practice run... ")
            play_fn(trial, practice_s, participant, f"practice_{trial.condition}", out_dir,
                    seed=protocol.PRACTICE_SEED)
        ask(f"Press Enter to start the {trial_s:.0f} s trial... ")
        result = play_fn(trial, trial_s, participant, trial.condition, out_dir)
        entry = {"position": trial.position, "condition": trial.condition, "seed": trial.seed,
                 **result}
        if result.get("end_reason") != "time":
            say(f"Run ended early ({result.get('end_reason')}); not counted. Rerun with --resume.")
            record["trials"].append(entry)
            save()
            raise SystemExit(1)
        entry["tlx"] = protocol.administer_tlx(ask, say)
        record["trials"].append(entry)
        save()
        say(f"Saved: score {result['score']}, crashes {result['crashes']}, "
            f"raw TLX {entry['tlx']['raw_tlx']}")

    record["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    save()
    say(f"\nSession complete: {out_path}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m study.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--participant", required=True, help="e.g. P07 (must end in a number)")
    p.add_argument("--session", type=int, required=True, choices=range(1, protocol.N_SESSIONS + 1))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--show", action="store_true", help="print the plan and exit")
    p.add_argument("--practice", type=float, default=protocol.PRACTICE_S, help="seconds; 0 to skip")
    p.add_argument("--duration", type=float, default=protocol.TRIAL_S, help="trial seconds")
    p.add_argument("--study-dir", type=Path, default=STUDY_DIR)
    a = p.parse_args(argv)

    if a.show:
        for t in protocol.schedule(a.participant, a.session):
            print(f"{t.position}. {t.condition:20s} input={t.input_mode:8s} seed={t.seed}")
        return
    run_session(a.participant, a.session, study_dir=a.study_dir, practice_s=a.practice,
                trial_s=a.duration, resume=a.resume)


if __name__ == "__main__":
    main()

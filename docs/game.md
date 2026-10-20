# The runner: implementation, logs, and latency

## Design

| Module | Role |
|---|---|
| `game/config.py` | Every tunable constant (`GameConfig`), logged with each session. |
| `game/obstacles.py` | Seeded obstacle generator. Rows are generated as a function of **distance**, never time, so one seed gives the identical course to every participant in every condition. Five row patterns force all four actions; no row is ever fully blocked (tested over 20 seeds). |
| `game/world.py` | Deterministic simulation in fixed `1/fps` steps with no pygame and no wall clock. Holds all game rules. |
| `game/app.py` | Main loop: gather input (keyboard / UDP / autopilot), apply, step, render, log. |
| `game/render.py` | Pseudo-3D pygame view; reads the world, never changes it. Shows a fading arrow for each received event (green = applied, grey = rejected). |
| `game/autopilot.py` | Simulated player + classifier with configurable latency, accuracy, and false-activation rate. |
| `game/replay.py`, `latency_report.py`, `sweep.py` | Analysis tools (below). |

**Rules.** Three lanes. Trains block a lane (change lanes). Low barriers need a
jump; overhead barriers need a roll. Jump and roll each last 0.65 s; rolling
mid-air drops you to the ground. Speed rises from 10 to 16 m/s.

**Modes.** `classic`: first crash ends the run, score = distance. `timed`
(use for the study): fixed duration, crashes are counted with a 1 s grace
period after each, so every condition gets identical exposure.

**Determinism.** Seed + the accepted inputs and the frames they landed on fully
determine a run. `python -m game.replay <log>` re-simulates and checks every
logged frame and the final summary; a mismatch means a log is incomplete or the
game code changed (each log records the git commit).

## Log format

`logs/<date>-<time>_<participant>_<input>_seed<N>.jsonl`, one JSON object per line.

| `type` | Fields |
|---|---|
| `session_start` | `wall_time`, `git`, `config` (full GameConfig), `input`, `participant`, `condition`, `port`, `autopilot` (if used) |
| `run_start` | `f`, `t`: countdown finished |
| `input` | `f` frame, `event` (the full message), `t_recv`, `accepted`; if rejected `reason` (`countdown`, `paused`, `over`, `low_confidence`); if accepted `effect` (`moved`, `blocked_edge`, `jumped`, `rolled`, `ignored_airborne`, `ignored_rolling`), `t_applied`, `t_displayed`. Autopilot inputs also carry `intended` (null = false activation). |
| `frame` | `f`, `d` distance, `v` speed, `lane`, `air`/`roll` frames left, `t` frame start, `t_disp` after flip |
| `spawn` | obstacle `id`, `row`, `lane`, `kind`, `start`, `length` |
| `crash` | `obstacle`, `kind`, `lane`, `state` (player snapshot) |
| `pass` | `obstacle`, `row`, `kind`, `lane`, `outcome`: `none` (never in its lane), `cleared` (jumped/rolled it), `hit`, `invulnerable` |
| `paused` / `running` | `f`, `t` |
| `end` / `session_end` | `frames`, `elapsed_s`, `distance`, `score`, `crashes`, `obstacles_passed`, `end_reason`; `session_end` adds `frame_ms` stats and `malformed_packets` |

## Latency instrumentation

Each accepted input carries a timestamp chain from one shared monotonic clock:

```
t_capture -> [classifier stages] -> timestamp (emit) -> t_recv -> t_applied -> t_displayed
 cap.read()   landmarks, inference   commit fired        game poll   world.apply   display.flip()
```

`python -m game.latency_report logs/*.jsonl` prints median / p90 / mean for
each segment, grouped by `source`, plus frame-timing statistics. Note that
`emit->recv` includes waiting for the game's next poll (the socket is drained
once per frame, so up to 16.7 ms at 60 fps), not only UDP transport.

**Not covered by software timestamps:** camera exposure and readout before
`cap.read()` returns, and display scan-out after `flip()`. To measure the true
end-to-end figure, film the hand and the screen together with a phone in
slow-motion (240 fps) and count frames from the start of the swipe to the lane
change on screen. Subtracting the logged `capture->displayed` gives the
camera + display overhead.

## Measurements so far (MacBook, Python 3.12, 60 fps)

**Game-side overhead** (UDP test with a simulated 120 ms classifier, 10 events):

| Segment | Median (ms) |
|---|---|
| emit -> recv (incl. wait for frame) | 14.0 |
| recv -> applied | 0.2 |
| applied -> displayed | 1.2 |
| frame interval p50 / p95 | 17.2 / 19.8, 0 late frames of 478 |

The game adds about one frame (~15 ms) on top of whatever the classifier takes.
Pipeline cost is dominated by the classifier side, as intended.

**Latency budget** (`python -m game.sweep`, autopilot, 10 seeds x 120 s, timed mode):

| Capture-to-action latency (ms) | 0-300 | 310-320 | >= 330 |
|---|---|---|---|
| Crashes / min, 100% accuracy | 0.00 | 10.6 | 17.1 |

The budget is a cliff at about 300 ms, set by the 0.65 s jump/roll window: the
simulated player times jumps as if input were instantaneous, so once latency
exceeds about half the window, every jump/roll lands after the barrier. Lane
changes keep working until the reaction lead (1 s). Real players anticipate, so
this is a conservative bound, and the online study measures the real one.
**Working target for the classifier: commit within ~250 ms of gesture onset,
including capture and landmark extraction.**

At 200 ms latency, accuracy costs more than false activations:

| Accuracy | 0 FA/min | 2 FA/min | 5 FA/min |
|---|---|---|---|
| 1.00 | 0.00 | 0.30 | 0.75 |
| 0.95 | 1.30 | 1.40 | 1.45 |
| 0.90 | 2.45 | 2.20 | 3.00 |
| 0.80 | 4.25 | 4.55 | 5.00 |

(crashes per minute, mean over 10 seeds). Every 5 points of accuracy is worth
about 1.2 crashes/min. Going from 0 to 5 false activations per minute costs
under 1. Early commits that trade a few points of accuracy for latency are only
worth it while latency is near or above the cliff.

**Still to measure:** `capture->displayed` with the real webcam pipeline
(`integration/threshold_bridge.py` with the game in `--input gesture`), and the
slow-motion camera check above.

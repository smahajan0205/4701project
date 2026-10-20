# Running study sessions

## Design (study/protocol.py)

- **Conditions:** `keyboard`, `gesture`, `gesture_constrained` (gesture with
  elbows kept at the sides).
- **Order:** fixed per participant and counterbalanced across participants. The six
  permutations are assigned by participant number (P01 -> order 1, ..., P07 -> order 1
  again). Any block of six consecutive participants puts every condition in every
  position, and after every other condition, exactly twice (tested).
- **Sessions:** three per participant, each running all conditions. Same order every session, so
  between-session change reflects learning.
- **Seeds:** seed = 100 + session number. Every participant sees the same course
  in every condition of a session, and a new course each session. Practice runs use seed 999.
- **Each condition:** 30 s practice (not analyzed), 120 s timed trial, then raw
  (unweighted) NASA-TLX, with six 0-100 subscales in steps of 5.
- **Consent** is confirmed by the experimenter at the start of session 1 and
  timestamped. Only game logs and TLX answers are stored; no video.

## Running

```bash
python -m study.run --participant P07 --session 1 --show   # print the plan
python -m study.run --participant P07 --session 1
```

Before gesture conditions, start the classifier in a second terminal (it sends
to UDP 5005). The harness prompts before each practice and trial, opens the game
window, and asks the TLX questions in the terminal after each trial; turn the
laptop to the participant for those.

If a run ends early (Esc, crash of the program), it is recorded as not counted
and the harness stops. `--resume` picks up at the first condition without a
completed trial.

## Output

```
logs/study/P07/session1.json                 order, seed, consent, git commit, per-trial results + TLX
logs/study/P07/<timestamp>_P07_<input>_seed101.jsonl   full game log per run (condition in header)
```

```bash
python -m study.summary                      # mean crashes / score / TLX by condition x session
python -m study.summary --csv results/study.csv
```

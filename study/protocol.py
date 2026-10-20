"""Study design: conditions, counterbalanced order, seeds, and NASA-TLX.

Each participant plays three sessions. A session runs every condition once, as
a timed run, followed by a NASA-TLX questionnaire. Condition order is fixed per
participant and counterbalanced across participants: the six permutations of
three conditions, assigned by participant number, put every condition in every
position (and after every other condition) equally often once six participants
have run. Keeping each participant's order the same across sessions means
session-to-session change reflects learning, not a reshuffled order.

Seeds depend only on the session number, so every participant faces the same
obstacle course in every condition of a given session, and a new course in
each later session.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

CONDITIONS = ("keyboard", "gesture", "gesture_constrained")
ORDERS = list(itertools.permutations(CONDITIONS))
N_SESSIONS = 3
TRIAL_S = 120.0
PRACTICE_S = 30.0
SEED_BASE = 100
PRACTICE_SEED = 999

# Game input mode and the instruction shown before each condition.
CONDITION_INFO = {
    "keyboard": ("keyboard", "Use the arrow keys: left/right change lanes, up jumps, down rolls."),
    "gesture": ("gesture", "Use hand swipes: left/right change lanes, up jumps, down rolls."),
    "gesture_constrained": ("gesture", "Use hand swipes, but keep both elbows touching your sides "
                                       "for the whole run."),
}


@dataclass(frozen=True)
class Trial:
    position: int     # 1-based position within the session
    condition: str
    input_mode: str
    seed: int


def participant_number(participant: str) -> int:
    """'P07' -> 7. IDs must end in a number so the order assignment is reproducible."""
    m = re.search(r"(\d+)$", participant)
    if not m:
        raise ValueError(f"participant id {participant!r} must end in a number, e.g. P07")
    return int(m.group(1))


def order_for(participant: str) -> tuple[str, ...]:
    return ORDERS[(participant_number(participant) - 1) % len(ORDERS)]


def session_seed(session: int) -> int:
    return SEED_BASE + session


def schedule(participant: str, session: int) -> list[Trial]:
    if not 1 <= session <= N_SESSIONS:
        raise ValueError(f"session must be 1..{N_SESSIONS}")
    return [Trial(i + 1, c, CONDITION_INFO[c][0], session_seed(session))
            for i, c in enumerate(order_for(participant))]


# ------------------------------------------------------------------ NASA-TLX

# Raw (unweighted) TLX: six 0-100 subscales in steps of 5, averaged.
TLX_SCALES = [
    ("mental", "Mental demand", "How mentally demanding was the task?", "very low", "very high"),
    ("physical", "Physical demand", "How physically demanding was the task?", "very low", "very high"),
    ("temporal", "Temporal demand", "How hurried or rushed was the pace of the task?",
     "very low", "very high"),
    ("performance", "Performance", "How successful were you in accomplishing what you were asked "
                                   "to do?", "perfect", "failure"),
    ("effort", "Effort", "How hard did you have to work to accomplish your level of performance?",
     "very low", "very high"),
    ("frustration", "Frustration", "How insecure, discouraged, irritated, stressed, and annoyed "
                                   "were you?", "very low", "very high"),
]


def parse_tlx(answer: str) -> int:
    """Accept 0-100, rounded to the nearest 5 (the TLX scale has 21 ticks)."""
    v = float(answer)
    if not 0 <= v <= 100:
        raise ValueError("must be between 0 and 100")
    return int(5 * round(v / 5))


def administer_tlx(ask=input, say=print) -> dict:
    say("\nNASA-TLX: rate each item from 0 to 100 (steps of 5).")
    out = {}
    for key, name, question, lo, hi in TLX_SCALES:
        while True:
            try:
                out[key] = parse_tlx(ask(f"  {name}: {question}\n    0 = {lo}, 100 = {hi} > "))
                break
            except ValueError as e:
                say(f"    {e}")
    out["raw_tlx"] = round(sum(out[k] for k, *_ in TLX_SCALES) / len(TLX_SCALES), 2)
    return out

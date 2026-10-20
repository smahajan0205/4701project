"""Re-simulate a logged session and check it reproduces exactly.

    python -m game.replay logs/<session>.jsonl            # verify
    python -m game.replay logs/<session>.jsonl --render   # verify, then watch it

Because the world is deterministic, the seed plus the accepted inputs (and the
frames they landed on) fully determine the run. A mismatch means the log is
incomplete or the game code changed since the session was recorded.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from common.events import EventType

from .config import GameConfig
from .logger import read_log
from .world import World

_COMPARE = ("frames", "distance", "score", "crashes", "obstacles_passed")


def load(path):
    records = read_log(path)
    start = records[0]
    assert start["type"] == "session_start", f"{path}: first record is not session_start"
    cfg = GameConfig.from_dict(start["config"])
    inputs: dict[int, list[EventType]] = defaultdict(list)
    frames: dict[int, dict] = {}
    end = None
    for r in records:
        if r["type"] == "input" and r.get("accepted"):
            inputs[r["f"]].append(EventType.parse(r["event"]["event_type"]))
        elif r["type"] == "frame":
            frames[r["f"]] = r
        elif r["type"] == "session_end":
            end = r
    return start, cfg, inputs, frames, end


def simulate(cfg: GameConfig, inputs, n_frames: int, frames=None, on_frame=None):
    """Replay; returns (world, first mismatching frame record or None)."""
    world = World(cfg)
    mismatch = None
    for f in range(n_frames):
        for a in inputs.get(f, ()):
            world.apply(a)
        world.step()
        if frames and mismatch is None and world.frame in frames:
            logged = frames[world.frame]
            snap = world.snapshot()
            if any(logged[k] != snap[k] for k in snap):
                mismatch = {"logged": {k: logged[k] for k in snap}, "replayed": snap}
        if on_frame:
            on_frame(world)
    return world, mismatch


def verify(path) -> bool:
    start, cfg, inputs, frames, end = load(path)
    if end is None:
        print(f"{path}: no session_end record (crashed or still running?)")
        return False
    world, mismatch = simulate(cfg, inputs, end["frames"], frames)
    got = world.summary()
    diffs = {k: (end.get(k), got.get(k)) for k in _COMPARE if end.get(k) != got.get(k)}
    ok = not diffs and mismatch is None
    n_in = sum(len(v) for v in inputs.values())
    print(f"{path}: {'OK' if ok else 'MISMATCH'}  ({end['frames']} frames, {n_in} inputs, "
          f"score {got['score']}, crashes {got['crashes']})")
    for k, (want, have) in diffs.items():
        print(f"  {k}: logged {want} replayed {have}")
    if mismatch:
        print(f"  first frame mismatch: {mismatch}")
    return ok


def render(path, speed: float = 1.0) -> None:
    import pygame

    from .render import H, W, Renderer

    start, cfg, inputs, _, end = load(path)
    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.SCALED | pygame.RESIZABLE)
    pygame.display.set_caption(f"Replay: {path}")
    renderer, clock = Renderer(screen), pygame.time.Clock()

    def on_frame(world):
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                raise SystemExit
        renderer.draw_world(world)
        renderer.draw_hud(world, f"replay {start['input']}", clock.get_fps(), None)
        pygame.display.flip()
        clock.tick(cfg.fps * speed)

    simulate(cfg, inputs, end["frames"], on_frame=on_frame)
    pygame.quit()


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m game.replay", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+")
    p.add_argument("--render", action="store_true")
    p.add_argument("--speed", type=float, default=1.0)
    a = p.parse_args(argv)
    ok = all([verify(path) for path in a.logs])
    if a.render:
        for path in a.logs:
            render(path, a.speed)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

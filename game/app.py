"""Main loop: collects input, steps the world, renders, and logs.

Timestamp chain for each input event (all from common.events.now()):
  t_capture    camera frame read            (set by the classifier, optional)
  timestamp    classifier committed/emitted (set by the classifier)
  t_recv       game pulled it off the socket / keyboard queue (once per frame)
  t_applied    game applied it to the world
  t_displayed  pygame.display.flip() returned for the frame showing its effect
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from common.events import DEFAULT_PORT, EventReceiver, EventType, GestureEvent, now

from .autopilot import Autopilot, AutopilotConfig
from .config import GameConfig
from .logger import LOG_DIR, SessionLogger, session_path
from .world import World

KEY_ACTIONS = {
    "left": EventType.LEFT, "a": EventType.LEFT,
    "right": EventType.RIGHT, "d": EventType.RIGHT,
    "up": EventType.JUMP, "w": EventType.JUMP, "space": EventType.JUMP,
    "down": EventType.ROLL, "s": EventType.ROLL,
}

INPUT_MODES = ("keyboard", "gesture", "both", "autopilot")


@dataclass
class AppOptions:
    game: GameConfig = field(default_factory=GameConfig)
    input: str = "keyboard"
    port: int = DEFAULT_PORT
    participant: str = "dev"
    condition: str = ""
    autopilot: AutopilotConfig = field(default_factory=AutopilotConfig)
    countdown_s: float = 3.0
    log: bool = True
    log_frames: bool = True
    log_dir: Path = LOG_DIR
    headless: bool = False
    exit_on_end: bool = False


def run(opts: AppOptions) -> list[Path]:
    """Play sessions until the user quits. Returns the log files written."""
    if opts.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    from .render import H, W, Renderer

    pygame.init()
    flags = 0 if opts.headless else pygame.SCALED | pygame.RESIZABLE
    screen = pygame.display.set_mode((W, H), flags)
    pygame.display.set_caption(f"4701 Runner [{opts.input}]")
    renderer = Renderer(screen)
    clock = pygame.time.Clock()
    receiver = EventReceiver(port=opts.port) if opts.input in ("gesture", "both") else None
    logs = []
    try:
        while True:
            outcome, path = _session(opts, pygame, screen, renderer, clock, receiver)
            if path:
                logs.append(path)
            if outcome == "quit":
                break
    finally:
        if receiver:
            receiver.close()
        pygame.quit()
    return logs


def _session(opts, pygame, screen, renderer, clock, receiver):
    cfg = opts.game
    fps = cfg.fps
    world = World(cfg)
    autopilot = Autopilot(world, opts.autopilot) if opts.input == "autopilot" else None
    logger = None
    if opts.log:
        header = {"config": cfg.to_dict(), "input": opts.input, "participant": opts.participant,
                  "condition": opts.condition, "port": opts.port}
        if autopilot:
            header["autopilot"] = opts.autopilot.to_dict()
        logger = SessionLogger(session_path(opts.participant, opts.input, cfg.seed, opts.log_dir),
                               header)
    if receiver:
        receiver.poll()  # discard anything sent before this session began

    state = "countdown" if opts.countdown_s > 0 else "running"
    countdown_left = round(opts.countdown_s * fps)
    feedback = None  # (event, accepted, t_displayed, latency_ms)
    frame_ms: list[float] = []
    last_disp = None
    keyboard_on = opts.input in ("keyboard", "both")

    def finish(reason: str):
        if world.end_reason is None:
            world.end_reason = reason
        if logger:
            summary = world.summary()
            summary["frame_ms"] = _frame_stats(frame_ms, fps)
            if receiver:
                summary["malformed_packets"] = receiver.malformed
            logger.close(summary)
        return logger.path if logger else None

    while True:
        t_start = now()
        received: list[tuple[GestureEvent, float, dict]] = []
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return "quit", finish("quit")
            if e.type != pygame.KEYDOWN:
                continue
            name = pygame.key.name(e.key)
            if state == "over" and name == "r":
                return "restart", finish("restart")
            if name == "p" and state in ("running", "paused"):
                state = "paused" if state == "running" else "running"
                if logger:
                    logger.write({"type": state, "f": world.frame, "t": now()})
            elif keyboard_on and name in KEY_ACTIONS:
                t = now()
                received.append((GestureEvent(KEY_ACTIONS[name], 1.0, t, source="keyboard"), t, {}))
        if receiver:
            received.extend((ev, t_recv, {}) for ev, t_recv in receiver.poll())
        if autopilot and state == "running":
            for emitted, intended, f_int in autopilot.poll():
                t_cap = t_start - (world.frame - f_int) / fps
                ev = GestureEvent(emitted, 1.0, t_start, t_capture=t_cap, source="autopilot")
                extra = {"intended": intended.value if intended else None}
                received.append((ev, t_start, extra))

        pending = []
        step_index = world.frame
        for ev, t_recv, extra in received:
            rec = {"type": "input", "f": step_index, "event": ev.to_dict(), "t_recv": t_recv, **extra}
            if state != "running":
                rec.update(accepted=False, reason=state)
            elif ev.confidence < cfg.min_confidence:
                rec.update(accepted=False, reason="low_confidence")
            else:
                rec.update(accepted=True, effect=world.apply(ev.event_type), t_applied=now())
            pending.append(rec)
            feedback = [ev, rec["accepted"], None, None]

        world_events = []
        if state == "running":
            world_events = world.step()
            if world.over:
                state = "over"
        elif state == "countdown":
            countdown_left -= 1
            if countdown_left <= 0:
                state = "running"
                if logger:
                    logger.write({"type": "run_start", "f": world.frame, "t": now()})

        renderer.draw_world(world)
        fb = None
        if feedback is not None:
            ev, accepted, t_disp, lat = feedback
            fb = (ev, accepted, 0.0 if t_disp is None else t_start - t_disp, lat)
        renderer.draw_hud(world, opts.input, clock.get_fps(), fb)
        if state == "countdown":
            renderer.draw_center(str(countdown_left // fps + 1), "get ready")
        elif state == "paused":
            renderer.draw_center("Paused", "P to resume")
        elif state == "over":
            s = world.summary()
            sub = f"score {s['score']}   crashes {s['crashes']}\nR to restart   Esc to quit"
            renderer.draw_center("Game over" if world.end_reason == "crash" else "Time!", sub)
        pygame.display.flip()
        t_disp = now()

        if feedback is not None and feedback[2] is None:
            ev = feedback[0]
            origin = ev.t_capture if ev.t_capture is not None else ev.timestamp
            feedback[2], feedback[3] = t_disp, (t_disp - origin) * 1000
        if logger:
            for rec in pending:
                rec["t_displayed"] = t_disp
                logger.write(rec)
            for e in world_events:
                logger.write({"f": step_index, **e})
            if opts.log_frames and step_index != world.frame:
                logger.write({"type": "frame", **world.snapshot(), "t": t_start, "t_disp": t_disp})
        if last_disp is not None and state == "running":
            frame_ms.append((t_disp - last_disp) * 1000)
        last_disp = t_disp

        if state == "over" and opts.exit_on_end:
            return "quit", finish(world.end_reason)
        clock.tick(fps)


def _frame_stats(frame_ms: list[float], fps: int) -> dict:
    if not frame_ms:
        return {}
    budget = 1000 / fps
    s = sorted(frame_ms)
    return {
        "n": len(s),
        "mean": round(statistics.fmean(s), 3),
        "p50": round(s[len(s) // 2], 3),
        "p95": round(s[int(len(s) * 0.95)], 3),
        "max": round(s[-1], 3),
        "late_frames": sum(x > 1.5 * budget for x in s),
    }

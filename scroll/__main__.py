"""Scrolling prototype: the same gesture events drive a document viewer.

    python -m scroll                              # keyboard: arrows (up/down scroll, left/right section)
    python -m scroll --input gesture              # classifier events on UDP 5005
    python -m scroll --input gesture --participant P03 --seed 2

Find each highlighted item; it counts once it is fully on screen. Logs go to
logs/scroll/ (target, input, acquired, session_end records).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from common.events import DEFAULT_PORT, EventReceiver, EventType, now
from game.logger import LOG_DIR, SessionLogger, session_path

from .task import ScrollConfig, ScrollTask

W = 520
MARGIN_TOP = 50
KEYS = {"up": EventType.JUMP, "down": EventType.ROLL, "left": EventType.LEFT, "right": EventType.RIGHT,
        "w": EventType.JUMP, "s": EventType.ROLL, "a": EventType.LEFT, "d": EventType.RIGHT}
BG, ITEM, ITEM_ALT, TARGET, TEXT, DIM = ((245, 245, 240), (255, 255, 255), (236, 238, 244),
                                         (255, 214, 90), (25, 25, 35), (110, 110, 120))


def draw(screen, fonts, task: ScrollTask, shown: float, last: str, input_label: str) -> None:
    cfg = task.cfg
    font, small = fonts
    screen.fill(BG)
    view = pygame.Rect(0, MARGIN_TOP, W, cfg.view_h)
    screen.set_clip(view)
    first = int(shown // cfg.item_h)
    for i in range(first, min(cfg.n_items, first + cfg.view_h // cfg.item_h + 2)):
        y = MARGIN_TOP + i * cfg.item_h - shown
        rect = pygame.Rect(16, y + 4, W - 32, cfg.item_h - 8)
        is_target = not task.done and i == task.target
        color = TARGET if is_target else (ITEM if (i // cfg.section_items) % 2 == 0 else ITEM_ALT)
        pygame.draw.rect(screen, color, rect, border_radius=8)
        section = i // cfg.section_items + 1
        screen.blit(font.render(f"Item {i + 1}", True, TEXT), (rect.x + 14, rect.y + 12))
        screen.blit(small.render(f"Section {section}" + ("   <- target" if is_target else ""),
                                 True, DIM), (rect.x + 14, rect.y + 44))
    screen.set_clip(None)
    status = (f"Done: {task.summary()['acquired']} targets. Esc to quit." if task.done else
              f"Find item {task.target + 1}  ({task.index + 1}/{len(task.targets)})")
    screen.blit(font.render(status, True, TEXT), (16, 12))
    screen.blit(small.render(f"[{input_label}] {last}", True, DIM), (W - 160, 16))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m scroll", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", choices=("keyboard", "gesture", "both"), default="keyboard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--targets", type=int, default=15)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--participant", default="dev")
    p.add_argument("--log-dir", type=Path, default=LOG_DIR / "scroll")
    p.add_argument("--exit-on-done", action="store_true", help="quit once every target is found")
    a = p.parse_args(argv)

    cfg = ScrollConfig(seed=a.seed, n_targets=a.targets)
    task = ScrollTask(cfg)
    pygame.init()
    screen = pygame.display.set_mode((W, cfg.view_h + MARGIN_TOP), pygame.SCALED | pygame.RESIZABLE)
    pygame.display.set_caption(f"4701 Scroll [{a.input}]")
    fonts = (pygame.font.SysFont("helvetica", 22), pygame.font.SysFont("helvetica", 15))
    clock = pygame.time.Clock()
    receiver = EventReceiver(port=a.port) if a.input in ("gesture", "both") else None
    logger = SessionLogger(session_path(a.participant, f"scroll-{a.input}", a.seed, a.log_dir),
                           {"config": cfg.to_dict(), "input": a.input, "participant": a.participant,
                            "min_confidence": a.min_confidence, "targets": task.targets})
    t0 = now()
    logger.write({"type": "target", "trial": 0, "target": task.target, "offset": 0,
                  "optimal": task.optimal, "t": t0})
    shown, last = 0.0, ""
    try:
        running = True
        while running:
            received = []
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
                elif e.type == pygame.KEYDOWN and a.input != "gesture":
                    name = pygame.key.name(e.key)
                    if name in KEYS:
                        received.append(({"event_type": KEYS[name].value, "confidence": 1.0,
                                          "source": "keyboard"}, KEYS[name], 1.0, now()))
            if receiver:
                received += [(ev.to_dict(), ev.event_type, ev.confidence, t) for ev, t in receiver.poll()]
            for ev_dict, action, conf, t_recv in received:
                rec = {"type": "input", "event": ev_dict, "t_recv": t_recv,
                       "accepted": conf >= a.min_confidence and not task.done}
                events = []
                if rec["accepted"]:
                    events = task.apply(action, now() - t0)
                    rec["offset"] = task.offset
                    last = action.value
                logger.write(rec)
                for e in events:
                    logger.write({**e, "t": now()})
            shown += (task.offset - shown) * 0.35  # ease toward the logical offset
            if abs(task.offset - shown) < 0.5:
                shown = float(task.offset)
            draw(screen, fonts, task, shown, last, a.input)
            pygame.display.flip()
            if task.done and a.exit_on_done:
                running = False
            clock.tick(60)
    finally:
        logger.close({"summary": task.summary()})
        if receiver:
            receiver.close()
        pygame.quit()
    print(task.summary())
    print(f"log: {logger.path}")


if __name__ == "__main__":
    main()

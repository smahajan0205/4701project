"""Pseudo-3D pygame rendering. Reads World state; never modifies it."""

from __future__ import annotations

import math

import pygame

from common.events import EventType

from .obstacles import Kind
from .world import World

W, H = 600, 760
HORIZON_Y = 170
GROUND_Y = 650       # screen y of the player's feet
CAM = 7.0            # perspective constant (m): larger = flatter
LANE_W_M = 2.6
PX_PER_M = 150 / LANE_W_M  # pixels per meter at the player's depth

SKY_TOP, SKY_BOTTOM = (120, 175, 230), (200, 225, 245)
GRASS = (110, 150, 90)
TRACK = (150, 130, 110)
SLEEPER = (115, 95, 80)
RAIL = (90, 90, 95)
TRAIN, TRAIN_FRONT, WINDOW = (70, 110, 160), (55, 85, 130), (190, 220, 240)
LOW_BAR, HIGH_BAR = (240, 200, 40), (210, 60, 50)
STRIPE, POST = (40, 40, 40), (90, 90, 90)
PLAYER, PLAYER_HEAD = (40, 90, 220), (240, 200, 160)
TEXT, TEXT_DIM = (20, 20, 30), (90, 90, 100)

ARROWS = {
    EventType.LEFT: [(-1, 0), (0, -0.6), (0, -0.25), (1, -0.25), (1, 0.25), (0, 0.25), (0, 0.6)],
}
ARROWS[EventType.RIGHT] = [(-x, y) for x, y in ARROWS[EventType.LEFT]]
ARROWS[EventType.JUMP] = [(y, x) for x, y in ARROWS[EventType.LEFT]]
ARROWS[EventType.ROLL] = [(y, -x) for x, y in ARROWS[EventType.LEFT]]


def project(x_m: float, z_m: float, h_m: float = 0.0) -> tuple[float, float, float]:
    """World (x across, z ahead of player, h up) in meters -> (screen x, y, scale)."""
    s = CAM / (CAM + max(z_m, -CAM + 0.5))
    y = HORIZON_Y + (GROUND_Y - HORIZON_Y) * s - h_m * PX_PER_M * s
    return W / 2 + x_m * PX_PER_M * s, y, s


def lane_x(lane: float) -> float:
    return (lane - 1) * LANE_W_M


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont("helvetica", 22)
        self.font_small = pygame.font.SysFont("helvetica", 16)
        self.font_big = pygame.font.SysFont("helvetica", 64, bold=True)
        self.sky = pygame.Surface((W, HORIZON_Y + 1))
        for y in range(HORIZON_Y + 1):
            t = y / HORIZON_Y
            c = [round(a + (b - a) * t) for a, b in zip(SKY_TOP, SKY_BOTTOM)]
            pygame.draw.line(self.sky, c, (0, y), (W, y))

    # ------------------------------------------------------------------ scene

    def draw_world(self, w: World) -> None:
        scr = self.screen
        scr.blit(self.sky, (0, 0))
        pygame.draw.rect(scr, GRASS, (0, HORIZON_Y, W, H - HORIZON_Y))
        view = w.cfg.view_distance_m
        edge = 1.5 * LANE_W_M + 0.3
        near, far = -2.5, view + 20
        poly = [project(-edge, near)[:2], project(edge, near)[:2],
                project(edge, far)[:2], project(-edge, far)[:2]]
        pygame.draw.polygon(scr, TRACK, poly)

        # Sleepers scroll with distance to give a sense of speed.
        spacing = 2.0
        z = -(w.distance % spacing) - spacing
        while z < view:
            if z > near:
                a, b = project(-edge, z), project(edge, z)
                pygame.draw.line(scr, SLEEPER, a[:2], b[:2], max(1, round(6 * a[2])))
            z += spacing
        for lane in range(3):
            for dx in (-0.6, 0.6):
                x = lane_x(lane) + dx
                pygame.draw.line(scr, RAIL, project(x, near)[:2], project(x, far)[:2], 2)

        ahead = sorted((o for o in w.obstacles if o.start - w.distance > 0.5),
                       key=lambda o: -o.start)
        beside = [o for o in w.obstacles if o.start - w.distance <= 0.5]
        for o in ahead:
            self._draw_obstacle(o, w.distance, view)
        self._draw_player(w)
        for o in beside:
            self._draw_obstacle(o, w.distance, view)

    def _draw_obstacle(self, o, distance: float, view: float) -> None:
        z0, z1 = o.start - distance, o.end - distance
        if z1 < -2.0 or z0 > view:
            return
        z0 = max(z0, -2.0)
        scr = self.screen
        cx = lane_x(o.lane)
        if o.kind is Kind.TRAIN:
            hw, h = 0.95, 3.2
            top = [project(cx - hw, z0, h), project(cx + hw, z0, h),
                   project(cx + hw, z1, h), project(cx - hw, z1, h)]
            pygame.draw.polygon(scr, TRAIN, [p[:2] for p in top])
            x0, y0, s = project(cx - hw, z0, h)
            x1, y1, _ = project(cx + hw, z0, 0)
            pygame.draw.rect(scr, TRAIN_FRONT, (x0, y0, x1 - x0, y1 - y0))
            wx0, wy0, _ = project(cx - hw * 0.75, z0, h * 0.85)
            wx1, wy1, _ = project(cx + hw * 0.75, z0, h * 0.55)
            pygame.draw.rect(scr, WINDOW, (wx0, wy0, wx1 - wx0, wy1 - wy0))
            return
        hw = 1.05
        lo, hi = (0.0, 0.9) if o.kind is Kind.LOW else (1.4, 2.3)
        color = LOW_BAR if o.kind is Kind.LOW else HIGH_BAR
        if o.kind is Kind.HIGH:
            for px in (cx - hw, cx + hw):
                a, b = project(px, z0, 0), project(px, z0, hi)
                pygame.draw.line(scr, POST, a[:2], b[:2], max(2, round(8 * a[2])))
        x0, y0, s = project(cx - hw, z0, hi)
        x1, y1, _ = project(cx + hw, z0, lo)
        rect = pygame.Rect(x0, y0, x1 - x0, y1 - y0)
        pygame.draw.rect(scr, color, rect)
        n = 5
        for i in range(n):
            sx = rect.x + rect.w * (i + 0.25) / n
            pygame.draw.line(scr, STRIPE, (sx, rect.bottom), (sx + rect.h * 0.6, rect.top),
                             max(1, round(5 * s)))

    def _draw_player(self, w: World) -> None:
        anim = w.cfg.frames(w.cfg.lane_anim_s)
        t = min(1.0, (w.frame - w.lane_change_frame) / anim)
        t = 1 - (1 - t) ** 2
        x = lane_x(w.prev_lane + (w.lane - w.prev_lane) * t)
        lift = 1.6 * math.sin(math.pi * w.jump_phase()) if w.airborne else 0.0
        height = 0.8 if w.rolling else 1.7
        sx, sy, s = project(x, 0.0)
        pygame.draw.ellipse(self.screen, (80, 70, 60),
                            (sx - 0.5 * PX_PER_M, sy - 8, PX_PER_M, 16))
        if w.invulnerable_left and (w.invulnerable_left // 6) % 2:
            return  # blink while invulnerable
        x0, y0, _ = project(x - 0.35, 0.0, lift + height)
        x1, y1, _ = project(x + 0.35, 0.0, lift)
        pygame.draw.rect(self.screen, PLAYER, (x0, y0, x1 - x0, y1 - y0), border_radius=8)
        hx, hy, _ = project(x, 0.0, lift + height + 0.25)
        pygame.draw.circle(self.screen, PLAYER_HEAD, (hx, hy), 0.28 * PX_PER_M)

    # -------------------------------------------------------------------- HUD

    def draw_hud(self, w: World, input_label: str, fps: float, feedback) -> None:
        scr = self.screen
        lines = [f"Score {w.score}", f"Speed {w.speed:4.1f} m/s"]
        if w.cfg.mode == "timed":
            lines.append(f"Time {max(0.0, w.cfg.duration_s - w.elapsed_s):5.1f} s")
            lines.append(f"Crashes {w.crashes}")
        for i, text in enumerate(lines):
            scr.blit(self.font.render(text, True, TEXT), (14, 10 + 26 * i))
        label = self.font_small.render(f"input: {input_label}   seed {w.cfg.seed}   {fps:4.0f} fps",
                                       True, TEXT_DIM)
        scr.blit(label, (W - label.get_width() - 12, 12))
        if feedback is not None:
            self._draw_feedback(*feedback)

    def _draw_feedback(self, event, accepted: bool, age_s: float, latency_ms) -> None:
        """Show the last received gesture so the player can see what the system saw."""
        if age_s > 0.8:
            return
        alpha = 1.0 - age_s / 0.8
        color = (120, 230, 140) if accepted else (180, 180, 180)
        cx, cy, r = W / 2 - 60, H - 52, 26
        pill = pygame.Surface((240, 70), pygame.SRCALPHA)
        pygame.draw.rect(pill, (20, 20, 30, round(170 * alpha)), pill.get_rect(), border_radius=35)
        self.screen.blit(pill, (cx - 45, cy - 35))
        pts = [(cx + r * px, cy + r * py) for px, py in ARROWS[event.event_type]]
        pygame.draw.polygon(self.screen, color, pts)
        text = f"{event.event_type.value} {event.confidence:.2f}"
        if latency_ms is not None:
            text += f"  {latency_ms:.0f} ms"
        surf = self.font_small.render(text, True, color)
        self.screen.blit(surf, (cx + r + 14, cy - surf.get_height() / 2))

    def draw_center(self, title: str, subtitle: str = "") -> None:
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((255, 255, 255, 140))
        self.screen.blit(shade, (0, 0))
        t = self.font_big.render(title, True, TEXT)
        self.screen.blit(t, (W / 2 - t.get_width() / 2, H / 2 - 90))
        for i, line in enumerate(subtitle.split("\n") if subtitle else []):
            s = self.font.render(line, True, TEXT)
            self.screen.blit(s, (W / 2 - s.get_width() / 2, H / 2 + 30 * i))

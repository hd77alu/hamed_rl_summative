"""2.5D isometric Pygame renderer for the FireShield environment.

Draws the terrain as raised isometric tiles (trees, houses, helipad, rocks,
lakes), animated flames with glow, water-soaked ground, firebreaks, a
quad-rotor drone with shadow and water-drop effect, and a HUD panel with
mission telemetry. Supports an on-screen window ("human") and off-screen
frames ("rgb_array") for videos and report figures.

Rendering never touches the environment's RNG, so it does not affect
reproducibility.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pygame

from environment.custom_env import (
    ACTION_NAMES,
    BASE,
    COMPASS,
    COMPASS_NAMES,
    DROP_WATER,
    FOREST,
    GRASS,
    ROCK,
    VILLAGE,
    WATER,
)

if TYPE_CHECKING:
    from environment.custom_env import FireShieldEnv

WIN_W, WIN_H = 1040, 640
HUD_W = 300
MAP_W = WIN_W - HUD_W
TILE_DEPTH = 8

BG_TOP = (18, 24, 38)
BG_BOTTOM = (40, 52, 70)
HUD_BG = (16, 20, 30)
TEXT = (230, 234, 240)
MUTED = (140, 150, 165)
ACCENT = (255, 140, 40)

GROUND = {
    GRASS: (128, 178, 72),
    FOREST: (58, 120, 52),
    ROCK: (128, 124, 118),
    WATER: (52, 120, 200),
    VILLAGE: (196, 176, 140),
    BASE: (88, 92, 104),
}
BURNT = (52, 46, 42)
FIREBREAK = (150, 110, 70)
SOAK = (70, 140, 230)


def _shade(color, k: float):
    return tuple(max(0, min(255, int(v * k))) for v in color)


def _mix(a, b, t: float):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class FireShieldRenderer:
    def __init__(self, env: "FireShieldEnv", mode: str = "human") -> None:
        self.env = env
        self.mode = mode
        self.frame = 0
        pygame.font.init()
        if mode == "human":
            pygame.display.init()
            pygame.display.set_caption("FireShield: Wildfire Response Drone")
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.Surface((WIN_W, WIN_H))
            self.clock = None
        self.font = self._font(15)
        self.font_small = self._font(12)
        self.font_big = self._font(24, bold=True)

        n = env.n
        self.tw = min(64, (MAP_W - 60) // n)
        self.th = self.tw // 2
        self.ox = MAP_W // 2
        self.oy = max(70, (WIN_H - n * self.th) // 2 - 10)
        self.background = self._make_background()
        self.glow = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

    @staticmethod
    def _font(size: int, bold: bool = False):
        try:
            return pygame.font.SysFont("segoeui,arial,helvetica", size, bold=bold)
        except Exception:  # pragma: no cover - font fallback
            return pygame.font.Font(None, size + 4)

    def _make_background(self) -> pygame.Surface:
        bg = pygame.Surface((WIN_W, WIN_H))
        for y in range(WIN_H):
            pygame.draw.line(bg, _mix(BG_TOP, BG_BOTTOM, y / WIN_H), (0, y), (WIN_W, y))
        return bg

    # ------------------------------------------------------------ geometry
    def _iso(self, r: float, c: float):
        """Top vertex of tile (r, c) in screen space."""
        return self.ox + (c - r) * self.tw / 2, self.oy + (c + r) * self.th / 2

    def _center(self, r: float, c: float):
        x, y = self._iso(r, c)
        return x, y + self.th / 2

    # ----------------------------------------------------------------- draw
    def draw(self):
        env = self.env
        self.frame += 1
        surf = self.screen
        surf.blit(self.background, (0, 0))
        self.glow.fill((0, 0, 0, 0))

        n = env.n
        for s in range(2 * n - 1):
            for r in range(max(0, s - n + 1), min(n, s + 1)):
                c = s - r
                self._draw_tile(surf, r, c)
                self._draw_object(surf, r, c)

        surf.blit(self.glow, (0, 0))
        self._draw_drone(surf)
        self._draw_hud(surf)

        if self.mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            pygame.display.flip()
            self.clock.tick(env.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)).copy()

    def _ground_color(self, r: int, c: int):
        env = self.env
        t = env.terrain[r, c]
        if env.burnt[r, c]:
            col = BURNT
        elif env.firebreak[r, c]:
            col = FIREBREAK
        else:
            col = GROUND[int(t)]
            if t in (GRASS, FOREST):
                col = _shade(col, 0.85 + 0.25 * float(env.fuel[r, c]))
        if env.wet[r, c] > 0 and t != WATER:
            col = _mix(col, SOAK, 0.45 * float(env.wet[r, c]))
        if env.intensity[r, c] > 0:
            col = _mix(col, (120, 40, 20), 0.5)
        return col

    def _draw_tile(self, surf, r: int, c: int) -> None:
        x, y = self._iso(r, c)
        hw, th, d = self.tw / 2, self.th, TILE_DEPTH
        if self.env.terrain[r, c] == WATER:
            d = 3
        col = self._ground_color(r, c)
        top = [(x, y), (x + hw, y + th / 2), (x, y + th), (x - hw, y + th / 2)]
        left = [(x - hw, y + th / 2), (x, y + th), (x, y + th + d), (x - hw, y + th / 2 + d)]
        right = [(x + hw, y + th / 2), (x, y + th), (x, y + th + d), (x + hw, y + th / 2 + d)]
        pygame.draw.polygon(surf, _shade(col, 0.62), left)
        pygame.draw.polygon(surf, _shade(col, 0.48), right)
        pygame.draw.polygon(surf, col, top)
        pygame.draw.polygon(surf, _shade(col, 0.8), top, 1)

    def _draw_object(self, surf, r: int, c: int) -> None:
        env = self.env
        cx, cy = self._center(r, c)
        t = int(env.terrain[r, c])
        u = self.tw / 56  # scale unit
        burning = env.intensity[r, c] > 0

        if env.burnt[r, c]:
            if t in (FOREST, VILLAGE):
                pygame.draw.line(surf, (25, 22, 20), (cx, cy), (cx, cy - 12 * u), max(2, int(3 * u)))
                pygame.draw.line(surf, (25, 22, 20), (cx, cy - 8 * u), (cx + 5 * u, cy - 13 * u), 2)
        elif env.firebreak[r, c]:
            for k in (-1, 0, 1):
                pygame.draw.line(
                    surf, (110, 78, 48), (cx - 12 * u + k * 6 * u, cy - 3 * u + k * 3 * u),
                    (cx + 4 * u + k * 6 * u, cy + 5 * u + k * 3 * u), 2,
                )
        elif t == FOREST:
            self._draw_tree(surf, cx - 6 * u, cy + 2 * u, 0.8 * u * (0.7 + 0.4 * env.fuel[r, c]), burning)
            self._draw_tree(surf, cx + 7 * u, cy - 2 * u, 0.65 * u * (0.7 + 0.4 * env.fuel[r, c]), burning)
        elif t == GRASS:
            green = (90, 150, 50) if not burning else (120, 90, 40)
            for dx, dy in ((-10, 2), (-2, -3), (7, 3), (12, -2)):
                bx, by = cx + dx * u, cy + dy * u
                pygame.draw.line(surf, green, (bx, by), (bx - 2 * u, by - 6 * u), 2)
                pygame.draw.line(surf, green, (bx, by), (bx + 2 * u, by - 6 * u), 2)
        elif t == ROCK:
            pts = [(cx - 12 * u, cy + 3 * u), (cx - 7 * u, cy - 7 * u), (cx + 3 * u, cy - 10 * u),
                   (cx + 12 * u, cy - 2 * u), (cx + 8 * u, cy + 5 * u)]
            pygame.draw.polygon(surf, (150, 146, 140), pts)
            pygame.draw.polygon(surf, (95, 92, 88), pts, 2)
        elif t == WATER:
            phase = self.frame * 0.15 + r + c
            for k in range(2):
                yy = cy - 2 * u + k * 5 * u + math.sin(phase + k) * 1.5
                pygame.draw.line(surf, (150, 200, 250), (cx - 8 * u, yy), (cx + 6 * u, yy), 1)
        elif t == VILLAGE:
            self._draw_house(surf, cx, cy, u)
        elif t == BASE:
            pygame.draw.ellipse(surf, (230, 230, 235), (cx - 16 * u, cy - 8 * u, 32 * u, 16 * u), 2)
            label = self.font_small.render("H", True, (240, 240, 245))
            surf.blit(label, label.get_rect(center=(cx, cy)))

        if burning:
            self._draw_fire(surf, r, c, cx, cy, float(env.intensity[r, c]), u)

    def _draw_tree(self, surf, x, y, s, burning: bool) -> None:
        pygame.draw.line(surf, (90, 60, 35), (x, y), (x, y - 8 * s), max(2, int(3 * s)))
        leaf = (30, 95, 40) if not burning else (95, 70, 30)
        leaf2 = (45, 125, 55) if not burning else (130, 90, 35)
        pygame.draw.polygon(surf, leaf, [(x - 10 * s, y - 6 * s), (x + 10 * s, y - 6 * s), (x, y - 24 * s)])
        pygame.draw.polygon(surf, leaf2, [(x - 7 * s, y - 15 * s), (x + 7 * s, y - 15 * s), (x, y - 30 * s)])

    def _draw_house(self, surf, cx, cy, u) -> None:
        w, h = 14 * u, 12 * u
        front = [(cx - w, cy - 2 * u), (cx, cy + 4 * u), (cx, cy + 4 * u - h), (cx - w, cy - 2 * u - h)]
        side = [(cx, cy + 4 * u), (cx + w, cy - 2 * u), (cx + w, cy - 2 * u - h), (cx, cy + 4 * u - h)]
        pygame.draw.polygon(surf, (225, 215, 195), front)
        pygame.draw.polygon(surf, (185, 172, 150), side)
        roof_l = [(cx - w - 2 * u, cy - 2 * u - h), (cx, cy + 4 * u - h), (cx, cy - 6 * u - 2 * h), (cx - w, cy - 10 * u - h)]
        roof_r = [(cx, cy + 4 * u - h), (cx + w + 2 * u, cy - 2 * u - h), (cx + w, cy - 10 * u - h), (cx, cy - 6 * u - 2 * h)]
        pygame.draw.polygon(surf, (178, 60, 45), roof_l)
        pygame.draw.polygon(surf, (140, 45, 35), roof_r)
        pygame.draw.rect(surf, (90, 70, 50), (cx - 8 * u, cy - 6 * u, 4 * u, 7 * u))

    def _draw_fire(self, surf, r, c, cx, cy, intensity, u) -> None:
        size = (10 + 18 * intensity) * u
        gw, gh = self.tw * 1.3, self.th * 1.3
        pygame.draw.ellipse(self.glow, (255, 120, 30, int(35 + 45 * intensity)), (cx - gw / 2, cy - gh / 2, gw, gh))
        for k, (col, scale, dx) in enumerate(
            (((200, 40, 20), 1.0, -5), ((255, 110, 20), 0.8, 4), ((255, 210, 70), 0.5, 0))
        ):
            flick = 0.85 + 0.25 * math.sin(self.frame * 0.9 + r * 3.1 + c * 1.7 + k * 2.0)
            hgt = size * scale * flick
            wid = size * scale * 0.45
            bx = cx + dx * u
            pygame.draw.polygon(
                surf, col,
                [(bx - wid, cy + 2 * u), (bx - wid * 0.4, cy - hgt * 0.55), (bx, cy - hgt),
                 (bx + wid * 0.4, cy - hgt * 0.55), (bx + wid, cy + 2 * u)],
            )

    def _draw_drone(self, surf) -> None:
        env = self.env
        r, c = env.drone
        cx, cy = self._center(r, c)
        u = self.tw / 56 * 1.3
        alt = 40 * u + 3 * math.sin(self.frame * 0.4)
        dy = cy - alt

        # Water drop effect on the 5-cell footprint
        if env.last_action == DROP_WATER and env.water_used > 0:
            for i, j in env._footprint(r, c):
                tx, ty = self._center(i, j)
                for k in range(5):
                    t = (k + self.frame % 3 / 3) / 5
                    px, py = cx + (tx - cx) * t, dy + (ty - dy) * t
                    pygame.draw.circle(surf, (120, 190, 255), (int(px), int(py)), max(2, int(3 * u)))

        pygame.draw.ellipse(surf, (20, 20, 20), (cx - 16 * u, cy - 4 * u, 32 * u, 10 * u))
        arm = 16 * u
        for ax, ay in ((-1, -0.5), (1, -0.5), (-1, 0.5), (1, 0.5)):
            ex, ey = cx + ax * arm, dy + ay * arm
            pygame.draw.line(surf, (60, 64, 72), (cx, dy), (ex, ey), max(2, int(3 * u)))
            spin = abs(math.sin(self.frame * 1.3 + ax + ay))
            rw = 18 * u * (0.4 + 0.6 * spin)
            pygame.draw.ellipse(surf, (200, 210, 225), (ex - rw / 2, ey - 3 * u, rw, 6 * u), 1)
        pygame.draw.ellipse(surf, (230, 90, 30), (cx - 10 * u, dy - 6 * u, 20 * u, 12 * u))
        # Tank gauge on the drone body
        frac = env.water / env.cfg.water_capacity
        pygame.draw.rect(surf, (30, 30, 40), (cx - 8 * u, dy + 7 * u, 16 * u, 4 * u))
        pygame.draw.rect(surf, (90, 170, 255), (cx - 8 * u, dy + 7 * u, 16 * u * frac, 4 * u))

    # ------------------------------------------------------------------ HUD
    def _bar(self, surf, x, y, w, frac, color, label, value) -> int:
        surf.blit(self.font.render(label, True, MUTED), (x, y))
        v = self.font.render(value, True, TEXT)
        surf.blit(v, (x + w - v.get_width(), y))
        pygame.draw.rect(surf, (45, 50, 62), (x, y + 22, w, 8), border_radius=4)
        pygame.draw.rect(surf, color, (x, y + 22, int(w * max(0.0, min(1.0, frac))), 8), border_radius=4)
        return y + 40

    def _draw_hud(self, surf) -> None:
        env, cfg = self.env, self.env.cfg
        x0 = MAP_W
        pygame.draw.rect(surf, HUD_BG, (x0, 0, HUD_W, WIN_H))
        pygame.draw.line(surf, (50, 56, 70), (x0, 0), (x0, WIN_H), 2)
        x, w = x0 + 20, HUD_W - 40
        y = 18
        surf.blit(self.font_big.render("FireShield", True, ACCENT), (x, y))
        y += 32
        surf.blit(self.font_small.render("Autonomous wildfire response drone", True, MUTED), (x, y))
        y += 28

        status_col = {"contained": (90, 200, 110), "village_burnt": (230, 70, 60),
                      "battery_depleted": (230, 160, 50), "timeout": (200, 200, 90)}.get(env.outcome, TEXT)
        surf.blit(self.font.render(f"Step {env.steps}/{cfg.max_steps}", True, TEXT), (x, y))
        st = self.font.render(env.outcome.replace("_", " ").upper(), True, status_col)
        surf.blit(st, (x + w - st.get_width(), y))
        y += 30

        y = self._bar(surf, x, y, w, env.water / cfg.water_capacity, (90, 170, 255), "Water tank", f"{env.water}/{cfg.water_capacity}")
        y = self._bar(surf, x, y, w, env.battery / cfg.battery_capacity, (120, 210, 120), "Battery", f"{env.battery}/{cfg.battery_capacity}")
        saved = env.forest_saved_fraction
        y = self._bar(surf, x, y, w, saved, (60, 160, 80), "Land saved", f"{saved * 100:.0f}%")
        y = self._bar(surf, x, y, w, env.dryness, (220, 150, 60), "Dryness", f"{env.dryness:.2f}")

        # Wind compass
        y += 4
        surf.blit(self.font.render("Wind", True, MUTED), (x, y))
        wcx, wcy, rad = x + w - 34, y + 30, 28
        pygame.draw.circle(surf, (45, 50, 62), (wcx, wcy), rad)
        pygame.draw.circle(surf, (80, 88, 104), (wcx, wcy), rad, 1)
        dr, dc = COMPASS[env.wind_dir]
        norm = math.hypot(dr, dc)
        # Convert grid direction into the isometric screen direction
        sx, sy = (dc - dr) / norm, (dc + dr) / norm / 2
        sn = math.hypot(sx, sy)
        sx, sy = sx / sn, sy / sn
        tip = (wcx + sx * (rad - 4), wcy + sy * (rad - 4))
        tail = (wcx - sx * (rad - 8), wcy - sy * (rad - 8))
        pygame.draw.line(surf, (180, 220, 255), tail, tip, 3)
        pygame.draw.circle(surf, (180, 220, 255), (int(tip[0]), int(tip[1])), 4)
        surf.blit(self.font.render(f"toward {COMPASS_NAMES[env.wind_dir]}", True, TEXT), (x, y + 22))
        surf.blit(self.font.render(f"speed {env.wind_speed:.2f}", True, TEXT), (x, y + 42))
        y += 74

        info = env._get_info()
        rows = [
            ("Active fires", str(info["fires_active"])),
            ("Cells burnt", str(info["cells_burnt"])),
            ("Extinguished", str(info["cells_extinguished"])),
            ("Firebreaks", str(info["firebreaks_built"])),
            ("Last action", ACTION_NAMES[env.last_action] if env.last_action is not None else "-"),
            ("Step reward", f"{env.last_reward:+.2f}"),
            ("Episode return", f"{env.episode_return:+.1f}"),
        ]
        for label, value in rows:
            surf.blit(self.font.render(label, True, MUTED), (x, y))
            v = self.font.render(value, True, TEXT)
            surf.blit(v, (x + w - v.get_width(), y))
            y += 22

        # Map title
        title = self.font.render(f"Wind-driven wildfire  |  {env.n} x {env.n} terrain", True, MUTED)
        surf.blit(title, (20, 16))

    def close(self) -> None:
        if self.mode == "human" and pygame.display.get_init():
            pygame.display.quit()

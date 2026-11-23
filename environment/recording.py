"""Record an MP4 of a policy playing FireShield (rendered with the rgb_array renderer).

    uv run main.py record                              # best model, test seeds 1000-1002
    uv run main.py record --algo dqn --episodes 5 --out assets/video/dqn.mp4
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # off-screen rendering

import pygame

from environment.custom_env import ACTION_NAMES, FireShieldEnv

FPS = 15
FRAMES_PER_STEP = 3  # 5 environment steps per second; extra frames animate flames/rotors
TITLE_BG, INK, MUTED = (18, 24, 33), (240, 239, 236), (170, 175, 185)
OUTCOME_COLOR = {"contained": (60, 200, 90), "village_burnt": (225, 70, 70),
                 "battery_depleted": (245, 180, 40), "timeout": (180, 180, 180)}


def _card(size: tuple[int, int], lines: list[tuple[str, int, tuple[int, int, int]]]) -> np.ndarray:
    pygame.font.init()
    surf = pygame.Surface(size)
    surf.fill(TITLE_BG)
    total = sum(s + 14 for _, s, _ in lines)
    y = (size[1] - total) // 2
    for text, fsize, color in lines:
        font = pygame.font.SysFont("segoeui,arial,helvetica", fsize, bold=fsize >= 28)
        img = font.render(text, True, color)
        surf.blit(img, ((size[0] - img.get_width()) // 2, y))
        y += fsize + 14
    return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))


def _banner(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    """Overlay an outcome banner across the bottom of the map area."""
    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    font = pygame.font.SysFont("segoeui,arial,helvetica", 30, bold=True)
    img = font.render(text, True, (255, 255, 255))
    pad = 14
    box = pygame.Surface((img.get_width() + 2 * pad, img.get_height() + pad), pygame.SRCALPHA)
    box.fill((*color, 225))
    x = (740 - box.get_width()) // 2
    surf.blit(box, (x, 28))
    surf.blit(img, (x + pad, 28 + pad // 2))
    return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))


def record(act: Callable[[np.ndarray, FireShieldEnv], int], label: str, out: Path, episodes: int = 3,
           seed: int = 1000, subtitle: str = "") -> dict:
    import imageio.v2 as imageio

    env = FireShieldEnv(render_mode="rgb_array")
    env.reset(seed=seed)
    size = env.render().shape[1::-1]
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=16,
                                ffmpeg_log_level="error")

    def hold(frame: np.ndarray, seconds: float) -> None:
        for _ in range(int(seconds * FPS)):
            writer.append_data(frame)

    results = []
    try:
        hold(_card(size, [("FireShield", 44, INK), ("Wildfire response drone", 22, MUTED),
                          (label, 22, INK), (subtitle, 16, MUTED),
                          (f"{episodes} unseen test episodes (seeds {seed}-{seed + episodes - 1})", 16, MUTED)]), 3.0)
        for ep in range(episodes):
            s = seed + ep
            obs, _ = env.reset(seed=s)
            env.action_space.seed(s)
            hold(_card(size, [(f"Episode {ep + 1}  -  seed {s}", 30, INK),
                              ("Goal: put out every fire before it reaches the village", 18, MUTED),
                              ("Drone: 4 water drops per tank, 80 battery; refill at the helipad", 16, MUTED)]), 1.6)
            hold(env.render(), 0.8)
            done, total, info = False, 0.0, {}
            while not done:
                a = act(obs, env)
                obs, r, term, trunc, info = env.step(a)
                total += r
                done = term or trunc
                for _ in range(FRAMES_PER_STEP):
                    writer.append_data(env.render())
            outcome = info["outcome"]
            text = f"{outcome.replace('_', ' ').upper()}  |  return {total:+.1f}  |  land saved {info['forest_saved']:.0%}"
            hold(_banner(env.render(), text, OUTCOME_COLOR[outcome]), 2.5)
            results.append({"seed": s, "outcome": outcome, "return": round(total, 1), "steps": info["steps"]})
            print(f"  episode {ep + 1} (seed {s}): {outcome:<16} return {total:7.1f} in {info['steps']} steps "
                  f"(last action {ACTION_NAMES[a]})")
    finally:
        writer.close()
        env.close()
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return {"file": str(out), "episodes": results}

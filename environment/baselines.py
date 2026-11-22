"""Non-learning reference policies used for sanity checks and report baselines."""

from __future__ import annotations

import numpy as np

from environment.custom_env import (
    DROP_WATER,
    MOVE_E,
    MOVE_N,
    MOVE_S,
    MOVE_W,
    RETURN_BASE,
    FireShieldEnv,
)


def random_policy(env: FireShieldEnv) -> int:
    return int(env.action_space.sample())


def greedy_policy(env: FireShieldEnv) -> int:
    """Hand-coded rule: refill when empty/low battery, drop on fire in reach,
    otherwise fly toward the fire cell closest to the village."""
    r, c = env.drone
    dist_base = abs(r - env.base_pos[0]) + abs(c - env.base_pos[1])
    if env.water == 0 or env.battery <= dist_base + 2:
        return RETURN_BASE
    if any(env.intensity[i, j] > 0 for i, j in env._footprint(r, c)):
        return DROP_WATER
    fires = np.argwhere(env.intensity > 0)
    if len(fires) == 0:
        return RETURN_BASE
    vill = env.village_cells
    threat = np.abs(fires[:, None, :] - vill[None, :, :]).sum(axis=2).min(axis=1)
    target = fires[int(np.argmin(threat))]
    dr, dc = target[0] - r, target[1] - c
    if abs(dr) >= abs(dc):
        return MOVE_S if dr > 0 else MOVE_N
    return MOVE_E if dc > 0 else MOVE_W

"""FireShield: a Gymnasium environment for an autonomous wildfire-response drone.

A single drone with a limited water tank and battery must contain a stochastic,
wind-driven wildfire on an N x N terrain grid and keep it away from a village.

Design summary (full specification in assignment.md section 4):
    Actions      Discrete(9): N, S, E, W, drop water, build firebreak,
                 return-to-base/refill, hover/scan, fly-to-nearest-fire
    Observation  flat float32 vector (local 5x5 fire + fuel maps, drone state,
                 weather, and global situational features)
    Terminated   fire fully out (success), village ignites (failure),
                 battery depleted (crash)
    Truncated    max_steps reached
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Terrain codes
GRASS, FOREST, ROCK, WATER, VILLAGE, BASE = range(6)
TERRAIN_NAMES = ("grass", "forest", "rock", "water", "village", "base")

# Actions
MOVE_N, MOVE_S, MOVE_E, MOVE_W, DROP_WATER, FIREBREAK, RETURN_BASE, HOVER, FLY_TO_FIRE = range(9)
ACTION_NAMES = (
    "move_north",
    "move_south",
    "move_east",
    "move_west",
    "drop_water",
    "build_firebreak",
    "return_to_base",
    "hover_scan",
    "fly_to_fire",
)
MOVE_DELTAS = {MOVE_N: (-1, 0), MOVE_S: (1, 0), MOVE_E: (0, 1), MOVE_W: (0, -1)}

# 8 compass directions as (d_row, d_col), clockwise from North. Wind index i
# means the wind blows TOWARD direction i (compass angle i * 45 degrees).
COMPASS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
COMPASS_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
_UNIT = np.array(COMPASS, np.float64) / np.linalg.norm(np.array(COMPASS, np.float64), axis=1, keepdims=True)
# WIND_COS[w, d] = cosine between wind direction w and spread direction d
WIND_COS = _UNIT @ _UNIT.T


@dataclass(frozen=True)
class FireShieldConfig:
    """All tunable environment parameters (kept in one place for reporting)."""

    grid_size: int = 12
    view_radius: int = 2  # local window is (2r+1) x (2r+1) = 5x5
    max_steps: int = 250
    # Drone
    water_capacity: int = 4  # number of drops per full tank
    battery_capacity: int = 80  # steps of flight per charge
    # Fire dynamics
    min_ignitions: int = 1
    max_ignitions: int = 3
    spread_prob: float = 0.10  # base ignition probability per burning neighbour
    diagonal_factor: float = 0.7  # diagonal neighbours spread less
    wind_strength: float = 1.5  # how strongly wind biases the spread
    ignition_intensity: float = 0.3
    intensity_growth: float = 0.15
    burn_rate: float = 0.04  # fuel consumed per step at intensity 1.0
    wet_decay: float = 0.1  # water-soaked cells dry out over ~10 steps
    wind_change_interval: int = 10
    dryness_range: tuple[float, float] = (0.7, 1.0)
    # Rewards
    r_ignite: float = -1.0  # each newly ignited cell (forest lost)
    r_extinguish: float = 0.5  # each burning cell put out
    r_village: float = -50.0  # village ignites -> episode ends
    r_win: float = 50.0  # fire fully out with village intact
    r_saved: float = 50.0  # bonus x fraction of flammable land saved
    r_wasted_drop: float = -0.5  # drop with empty tank or onto no fire
    r_firebreak: float = -0.2  # firebreak cost (pays off later)
    r_invalid: float = -0.5  # firebreak on an invalid cell
    r_bump: float = -0.1  # trying to fly off the map
    r_step: float = -0.05  # time cost
    r_crash: float = -20.0  # battery depleted
    # Potential-based shaping (Ng et al., 1999): F = gamma * phi(s') - phi(s),
    # phi = -coef * distance(drone, target) / max_distance, where the target is the
    # nearest fire while the tank has water, otherwise the base. This leaves the
    # optimal policy unchanged and gives a dense signal. 0 disables it.
    shaping_coef: float = 3.0
    shaping_gamma: float = 0.99


class FireShieldEnv(gym.Env):
    """Wildfire containment with a water-carrying drone."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        render_mode: str | None = None,
        config: FireShieldConfig | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        cfg = config or FireShieldConfig()
        if overrides:
            valid = {f.name for f in fields(FireShieldConfig)}
            unknown = set(overrides) - valid
            if unknown:
                raise ValueError(f"Unknown config keys: {sorted(unknown)}")
            cfg = replace(cfg, **overrides)
        self.cfg = cfg
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode

        n, k = cfg.grid_size, 2 * cfg.view_radius + 1
        self.n = n
        self.window = k
        self.action_space = spaces.Discrete(len(ACTION_NAMES))

        # Observation layout: (name, size, low, high). Kept as data so the
        # report's provenance table and tests can be generated from it.
        self.obs_layout: list[tuple[str, int, float, float]] = [
            ("local_fire", k * k, 0.0, 1.0),
            ("local_fuel", k * k, 0.0, 1.0),
            ("drone_pos", 2, 0.0, 1.0),
            ("water_level", 1, 0.0, 1.0),
            ("battery_level", 1, 0.0, 1.0),
            ("wind_dir_sincos", 2, -1.0, 1.0),
            ("wind_speed", 1, 0.0, 1.0),
            ("dryness", 1, 0.0, 1.0),
            ("vec_nearest_fire", 2, -1.0, 1.0),
            ("vec_base", 2, -1.0, 1.0),
            ("village_threat", 1, 0.0, 1.0),
            ("frac_burning", 1, 0.0, 1.0),
            ("frac_burnt", 1, 0.0, 1.0),
        ]
        low = np.concatenate([np.full(s, lo, np.float32) for _, s, lo, _ in self.obs_layout])
        high = np.concatenate([np.full(s, hi, np.float32) for _, s, _, hi in self.obs_layout])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._spread_buf = np.zeros((n + 2, n + 2), np.float32)
        self._window_buf = np.zeros((n + 2 * cfg.view_radius, n + 2 * cfg.view_radius), np.float32)
        self._renderer = None
        self.last_action: int | None = None
        self.last_reward = 0.0
        self.episode_return = 0.0

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg, n, rng = self.cfg, self.n, self.np_random

        self._generate_terrain()

        self.intensity = np.zeros((n, n), np.float32)  # 0 = not burning
        self.burnt = np.zeros((n, n), bool)
        self.firebreak = np.zeros((n, n), bool)
        self.wet = np.zeros((n, n), np.float32)

        # Weather
        self.wind_dir = int(rng.integers(8))
        self.wind_speed = float(rng.uniform(0.2, 0.9))
        self.dryness = float(rng.uniform(*cfg.dryness_range))

        # Ignitions: flammable cells far from the village
        vr, vc = self.village_center
        rows, cols = np.indices((n, n))
        far = np.maximum(abs(rows - vr), abs(cols - vc)) >= n // 2
        candidates = np.argwhere(far & (self.fuel > 0) & (self.terrain != BASE))
        count = int(rng.integers(cfg.min_ignitions, cfg.max_ignitions + 1))
        picks = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
        for r, c in candidates[picks]:
            self.intensity[r, c] = cfg.ignition_intensity

        self.initial_flammable = int((self.fuel > 0).sum())

        # Drone
        self.drone = np.array(self.base_pos, dtype=int)
        self.water = cfg.water_capacity
        self.battery = cfg.battery_capacity

        # Episode bookkeeping
        self.steps = 0
        self.water_used = 0
        self.cells_extinguished = 0
        self.firebreaks_built = 0
        self.village_burnt = False
        self.outcome = "running"
        self.last_action = None
        self.last_reward = 0.0
        self.episode_return = 0.0
        self.episode_shaping = 0.0

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def _generate_terrain(self) -> None:
        """Smoothed-noise fuel map with grass, forest, rock and water patches."""
        n, rng = self.n, self.np_random
        noise = rng.random((n, n))
        for _ in range(2):  # 3x3 box blur twice -> spatially coherent patches
            p = np.pad(noise, 1, mode="edge")
            noise = sum(p[i : i + n, j : j + n] for i in range(3) for j in range(3)) / 9.0
        q10, q35 = np.quantile(noise, [0.10, 0.40])

        terrain = np.full((n, n), FOREST, np.int8)
        terrain[noise < q35] = GRASS
        low_cells = noise < q10
        terrain[low_cells] = np.where(rng.random((n, n)) < 0.5, WATER, ROCK)[low_cells]

        # Village: 2x2 block in a random corner; base 4 cells toward the centre
        corner = int(rng.integers(4))
        vr = 0 if corner in (0, 1) else n - 2
        vc = 0 if corner in (0, 2) else n - 2
        terrain[vr : vr + 2, vc : vc + 2] = VILLAGE
        self.village_cells = np.array([(vr + i, vc + j) for i in range(2) for j in range(2)])
        self.village_center = (vr + 0.5, vc + 0.5)

        step_r = 1 if vr == 0 else -1
        step_c = 1 if vc == 0 else -1
        br = vr + (2 if vr == 0 else -1) + step_r * 2
        bc = vc + (2 if vc == 0 else -1) + step_c * 2
        terrain[br, bc] = BASE
        self.base_pos = (br, bc)

        fuel = np.zeros((n, n), np.float32)
        fuel[terrain == GRASS] = rng.uniform(0.3, 0.6, (terrain == GRASS).sum())
        fuel[terrain == FOREST] = rng.uniform(0.6, 1.0, (terrain == FOREST).sum())
        fuel[terrain == VILLAGE] = 0.8
        self.terrain = terrain
        self.fuel = fuel

    # ------------------------------------------------------------------- step
    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        cfg = self.cfg
        reward = cfg.r_step
        r, c = self.drone
        phi_before = self._potential()

        # 1) Apply the drone action
        if action in MOVE_DELTAS:
            reward += self._move(*MOVE_DELTAS[action])
        elif action == DROP_WATER:
            reward += self._drop_water(r, c)
        elif action == FIREBREAK:
            reward += self._build_firebreak(r, c)
        elif action == RETURN_BASE:
            if (r, c) == self.base_pos:
                self.water = cfg.water_capacity
                self.battery = cfg.battery_capacity
            else:
                reward += self._step_toward(self.base_pos)
        elif action == FLY_TO_FIRE:
            # waypoint autopilot: one step toward the nearest burning cell
            target = self._nearest_fire()
            if target is not None and tuple(target) != (r, c):
                reward += self._step_toward(target)
        # HOVER: no-op (holding pattern / scanning)

        # 2) Advance the fire simulation
        newly_ignited = self._advance_fire()
        reward += cfg.r_ignite * float(newly_ignited.sum())

        # 3) Drone systems
        self.steps += 1
        on_base = tuple(self.drone) == self.base_pos
        if not (action == RETURN_BASE and on_base):
            self.battery -= 1

        # 4) Termination checks
        terminated = False
        if newly_ignited[self.terrain == VILLAGE].any():
            self.village_burnt = True
            reward += cfg.r_village
            terminated, self.outcome = True, "village_burnt"
        elif not (self.intensity > 0).any():
            reward += cfg.r_win + cfg.r_saved * self.forest_saved_fraction
            terminated, self.outcome = True, "contained"
        elif self.battery <= 0:
            reward += cfg.r_crash
            terminated, self.outcome = True, "battery_depleted"

        truncated = not terminated and self.steps >= cfg.max_steps
        if truncated:
            self.outcome = "timeout"

        if cfg.shaping_coef:
            phi_after = 0.0 if terminated else self._potential()
            shaping = cfg.shaping_gamma * phi_after - phi_before
            reward += shaping
            self.episode_shaping += shaping

        self.last_action = action
        self.last_reward = reward
        self.episode_return += reward
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _nearest_fire(self) -> np.ndarray | None:
        fires = np.argwhere(self.intensity > 0)
        if len(fires) == 0:
            return None
        return fires[int(np.argmin(np.abs(fires - self.drone).sum(axis=1)))]

    def _step_toward(self, target) -> float:
        dr, dc = int(target[0]) - self.drone[0], int(target[1]) - self.drone[1]
        if abs(dr) >= abs(dc):
            return self._move(int(np.sign(dr)), 0)
        return self._move(0, int(np.sign(dc)))

    def _potential(self) -> float:
        if not self.cfg.shaping_coef:
            return 0.0
        if self.water > 0:
            fires = np.argwhere(self.intensity > 0)
            if len(fires) == 0:
                return 0.0
            dist = float(np.abs(fires - self.drone).sum(axis=1).min())
        else:
            dist = float(abs(self.drone[0] - self.base_pos[0]) + abs(self.drone[1] - self.base_pos[1]))
        return -self.cfg.shaping_coef * dist / (2 * (self.n - 1))

    def _move(self, dr: int, dc: int) -> float:
        nr, nc = self.drone[0] + dr, self.drone[1] + dc
        if 0 <= nr < self.n and 0 <= nc < self.n:
            self.drone[:] = (nr, nc)
            return 0.0
        return self.cfg.r_bump

    def _footprint(self, r: int, c: int) -> list[tuple[int, int]]:
        cells = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [(i, j) for i, j in cells if 0 <= i < self.n and 0 <= j < self.n]

    def _drop_water(self, r: int, c: int) -> float:
        cfg = self.cfg
        if self.water <= 0:
            return cfg.r_wasted_drop
        self.water -= 1
        self.water_used += 1
        put_out = 0
        for i, j in self._footprint(r, c):
            if self.intensity[i, j] > 0:
                self.intensity[i, j] = 0.0
                put_out += 1
            self.wet[i, j] = 1.0
        self.cells_extinguished += put_out
        if put_out == 0:
            return cfg.r_wasted_drop
        return cfg.r_extinguish * put_out

    def _build_firebreak(self, r: int, c: int) -> float:
        cfg = self.cfg
        valid = (
            self.fuel[r, c] > 0
            and self.intensity[r, c] == 0
            and not self.burnt[r, c]
            and self.terrain[r, c] not in (VILLAGE, BASE)
        )
        if not valid:
            return cfg.r_invalid
        self.firebreak[r, c] = True
        self.fuel[r, c] = 0.0
        self.firebreaks_built += 1
        return cfg.r_firebreak

    def _advance_fire(self) -> np.ndarray:
        """One tick of burn-down + wind-driven stochastic spread.

        Returns a boolean mask of cells that ignited this tick.
        """
        cfg, n = self.cfg, self.n
        burning = self.intensity > 0

        # Burn-down: intensity grows, fuel is consumed, exhausted cells burn out
        self.intensity[burning] = np.minimum(1.0, self.intensity[burning] + cfg.intensity_growth)
        self.fuel[burning] = np.maximum(0.0, self.fuel[burning] - cfg.burn_rate * self.intensity[burning])
        exhausted = burning & (self.fuel <= 0)
        self.intensity[exhausted] = 0.0
        self.burnt[exhausted] = True

        # Spread: P(ignite) = 1 - prod_neighbours(1 - p_dir)
        src = self._spread_buf  # zero border, interior = current intensity
        src[1:-1, 1:-1] = self.intensity
        no_ignite = np.ones((n, n), np.float32)
        cos_row = WIND_COS[self.wind_dir]
        for d, (dr, dc) in enumerate(COMPASS):
            wind_factor = max(0.1, 1.0 + cfg.wind_strength * self.wind_speed * cos_row[d])
            dist_factor = cfg.diagonal_factor if dr and dc else 1.0
            # Fire spreading in direction (dr, dc) reaches [r, c] from [r-dr, c-dc]
            shifted = src[1 - dr : 1 - dr + n, 1 - dc : 1 - dc + n]
            no_ignite *= 1.0 - (cfg.spread_prob * dist_factor * wind_factor) * shifted
        target_factor = self.fuel * self.dryness * (1.0 - self.wet)
        p_ignite = (1.0 - no_ignite) * target_factor
        susceptible = (self.fuel > 0) & (self.intensity == 0) & ~self.burnt
        ignited = susceptible & (self.np_random.random((n, n)) < p_ignite)
        self.intensity[ignited] = cfg.ignition_intensity

        # Weather drift
        self.wet = np.maximum(0.0, self.wet - cfg.wet_decay)
        if (self.steps + 1) % cfg.wind_change_interval == 0:
            self.wind_dir = (self.wind_dir + int(self.np_random.integers(-1, 2))) % 8
            self.wind_speed = float(np.clip(self.wind_speed + self.np_random.uniform(-0.1, 0.1), 0.0, 1.0))
        return ignited

    # ---------------------------------------------------------- observations
    @property
    def forest_saved_fraction(self) -> float:
        lost = int((self.burnt | (self.intensity > 0)).sum())
        return max(0.0, 1.0 - lost / max(1, self.initial_flammable))

    def _local_window(self, grid: np.ndarray) -> np.ndarray:
        rad = self.cfg.view_radius
        padded = self._window_buf  # out-of-map cells read as 0 (no fire / no fuel)
        padded[rad:-rad, rad:-rad] = grid
        r, c = self.drone
        return padded[r : r + 2 * rad + 1, c : c + 2 * rad + 1].ravel()

    def _get_obs(self) -> np.ndarray:
        n = self.n
        scale = max(1, n - 1)
        r, c = self.drone
        fire_cells = np.argwhere(self.intensity > 0)
        if len(fire_cells):
            d = np.abs(fire_cells - self.drone).sum(axis=1)
            nearest = fire_cells[int(np.argmin(d))]
            vec_fire = (nearest - self.drone) / scale
            vill_dist = np.max(
                np.abs(fire_cells[:, None, :] - self.village_cells[None, :, :]), axis=2
            ).min()
            threat = 1.0 - min(1.0, vill_dist / scale)
        else:
            vec_fire = np.zeros(2)
            threat = 0.0
        angle = self.wind_dir * np.pi / 4
        obs = np.concatenate(
            [
                self._local_window(self.intensity),
                self._local_window(self.fuel),
                [r / scale, c / scale],
                [self.water / self.cfg.water_capacity],
                [self.battery / self.cfg.battery_capacity],
                [np.sin(angle), np.cos(angle)],
                [self.wind_speed],
                [self.dryness],
                vec_fire,
                (np.array(self.base_pos) - self.drone) / scale,
                [threat],
                [(self.intensity > 0).mean()],
                [self.burnt.mean()],
            ]
        ).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "forest_saved": self.forest_saved_fraction,
            "village_burnt": self.village_burnt,
            "fires_active": int((self.intensity > 0).sum()),
            "cells_burnt": int(self.burnt.sum()),
            "cells_extinguished": self.cells_extinguished,
            "water_used": self.water_used,
            "firebreaks_built": self.firebreaks_built,
            "water": self.water,
            "battery": self.battery,
            "outcome": self.outcome,
            "episode_shaping": getattr(self, "episode_shaping", 0.0),
        }

    # -------------------------------------------------------------- rendering
    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode; pass render_mode='human' or 'rgb_array'.")
            return None
        if self._renderer is None:
            from environment.rendering import FireShieldRenderer

            self._renderer = FireShieldRenderer(self, mode=self.render_mode)
        return self._renderer.draw()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

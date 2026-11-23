"""Deployable FireShield policy: load a trained model and turn drone state into an action.

Three ways to load the same policy (the serialization path):

* ``FireShieldPolicy.from_checkpoint("models/pg/ppo_best.pt")``  - PyTorch training checkpoint
* ``FireShieldPolicy.from_torchscript("models/export/policy.torchscript.pt")`` - no project code needed
* ``FireShieldPolicy.from_numpy("models/export/policy_weights.npz")``  - NumPy only, no PyTorch
  (what an embedded flight computer or a mobile/web frontend would ship)

Two kinds of input:

* ``act(observation)`` - the 65-number observation vector the agent was trained on
* ``act_from_state(state)`` - a JSON-style world state (drone position, water, battery, wind,
  fire/fuel grids...). It is converted with the *environment's own* observation code, so a
  real controller and the simulator always produce identical observations.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from environment.custom_env import ACTION_NAMES, COMPASS_NAMES, FireShieldEnv

ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "models" / "export"


class StateError(ValueError):
    """The world state sent by a client is malformed."""


# ----------------------------------------------------------------- state -> obs
_ENV: FireShieldEnv | None = None


def _template_env() -> FireShieldEnv:
    global _ENV
    if _ENV is None:
        _ENV = FireShieldEnv()
        _ENV.reset(seed=0)
    return _ENV


def _grid(state: dict, key: str, n: int, default: float | None = None) -> np.ndarray:
    if key not in state:
        if default is None:
            raise StateError(f"missing field '{key}'")
        return np.full((n, n), default, np.float32)
    arr = np.asarray(state[key], dtype=np.float32)
    if arr.shape != (n, n):
        raise StateError(f"'{key}' must be a {n}x{n} grid, got shape {arr.shape}")
    return arr


def _cell(value: Any, name: str, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=int)
    if arr.shape != (2,) or not ((0 <= arr) & (arr < n)).all():
        raise StateError(f"'{name}' must be [row, col] inside the {n}x{n} map, got {value}")
    return arr


def observation_from_state(state: dict[str, Any]) -> np.ndarray:
    """Build the 65-dim observation from a world-state dict (see ``example_state``)."""
    env = _template_env()
    n, cfg = env.n, env.cfg
    try:
        drone = _cell(state["drone"], "drone", n)
        base = _cell(state["base"], "base", n)
        village = np.asarray(state["village_cells"], dtype=int).reshape(-1, 2)
        wind = state["wind"]
        direction = wind["direction"]
        wind_dir = COMPASS_NAMES.index(direction) if isinstance(direction, str) else int(direction)
        water, battery = int(state["water"]), int(state["battery"])
    except KeyError as e:
        raise StateError(f"missing field {e}") from None
    except (TypeError, ValueError) as e:
        raise StateError(str(e)) from None
    if not 0 <= wind_dir < 8:
        raise StateError(f"wind direction must be one of {COMPASS_NAMES} or 0-7")
    if not (0 <= water <= cfg.water_capacity and 0 <= battery <= cfg.battery_capacity):
        raise StateError(f"water must be 0-{cfg.water_capacity} and battery 0-{cfg.battery_capacity}")

    env.intensity = _grid(state, "fire_intensity", n)
    env.fuel = _grid(state, "fuel", n)
    env.burnt = _grid(state, "burnt", n, default=0.0).astype(bool)
    env.drone = drone
    env.base_pos = (int(base[0]), int(base[1]))
    env.village_cells = village
    env.water, env.battery = water, battery
    env.wind_dir, env.wind_speed = wind_dir, float(wind.get("speed", 0.5))
    env.dryness = float(state.get("dryness", 0.85))
    return env._get_obs()


def state_from_env(env: FireShieldEnv) -> dict[str, Any]:
    """Serialize a live simulator into the JSON world state (what a drone would report)."""
    return {
        "drone": [int(v) for v in env.drone],
        "base": list(env.base_pos),
        "village_cells": env.village_cells.tolist(),
        "water": int(env.water),
        "battery": int(env.battery),
        "wind": {"direction": COMPASS_NAMES[env.wind_dir], "speed": float(env.wind_speed)},
        "dryness": float(env.dryness),
        "fire_intensity": env.intensity.tolist(),
        "fuel": env.fuel.tolist(),
        "burnt": env.burnt.astype(int).tolist(),
    }


def example_state(seed: int = 1000) -> dict[str, Any]:
    env = FireShieldEnv()
    env.reset(seed=seed)
    return state_from_env(env)


# ----------------------------------------------------------------- policy
def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


class FireShieldPolicy:
    """Framework-agnostic wrapper: ``forward`` maps an observation to raw outputs."""

    def __init__(self, forward: Callable[[np.ndarray], np.ndarray], meta: dict[str, Any]):
        self._forward = forward
        self.meta = meta
        self.obs_dim = int(meta["obs_dim"])
        self.output = meta["output"]  # "logits" (policy gradient) or "q_values" (DQN)

    # loaders --------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str | Path) -> FireShieldPolicy:
        import torch

        from training.common import load_policy

        net, payload = load_policy(path)

        def forward(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return net(torch.as_tensor(obs, dtype=torch.float32)).numpy()

        meta = {k: payload[k] for k in ("algo", "obs_dim", "n_actions", "hidden", "depth", "output")}
        meta["source"] = str(path)
        return cls(forward, meta)

    @classmethod
    def from_torchscript(cls, path: str | Path, card: str | Path | None = None) -> FireShieldPolicy:
        import torch

        net = torch.jit.load(str(path), map_location="cpu").eval()

        def forward(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return net(torch.as_tensor(obs, dtype=torch.float32)).numpy()

        meta = _read_card(card or Path(path).with_name("policy_card.json"))
        meta["source"] = str(path)
        return cls(forward, meta)

    @classmethod
    def from_numpy(cls, path: str | Path, card: str | Path | None = None) -> FireShieldPolicy:
        data = np.load(path)
        n_layers = len([k for k in data.files if k.startswith("W")])
        weights = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n_layers)]

        def forward(obs: np.ndarray) -> np.ndarray:
            x = np.asarray(obs, dtype=np.float32)
            for i, (W, b) in enumerate(weights):
                x = x @ W.T + b
                if i < n_layers - 1:
                    x = np.maximum(x, 0.0)  # ReLU
            return x

        meta = _read_card(card or Path(path).with_name("policy_card.json"))
        meta["source"] = str(path)
        return cls(forward, meta)

    # inference ------------------------------------------------------
    def act(self, observation: Sequence[float] | np.ndarray) -> dict[str, Any]:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise StateError(f"observation must have {self.obs_dim} values, got shape {obs.shape}")
        if not np.isfinite(obs).all():
            raise StateError("observation contains NaN or infinity")
        out = self._forward(obs)
        action = int(np.argmax(out))
        result: dict[str, Any] = {"action": action, "action_name": ACTION_NAMES[action]}
        if self.output == "logits":
            probs = _softmax(out)
            result["confidence"] = round(float(probs[action]), 4)
            result["probabilities"] = {ACTION_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)}
        else:
            result["q_values"] = {ACTION_NAMES[i]: round(float(q), 4) for i, q in enumerate(out)}
        return result

    def act_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return self.act(observation_from_state(state))


def _read_card(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

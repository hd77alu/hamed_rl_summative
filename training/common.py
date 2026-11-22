"""Shared training infrastructure: env factory, networks, evaluation, logging,
checkpointing. Every algorithm uses these so the comparison is fair."""

from __future__ import annotations

import csv
import json
import logging
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from environment.baselines import greedy_policy, random_policy
from environment.custom_env import FireShieldEnv

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("torchrl").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models"

TOTAL_FRAMES = 150_000
VAL_SEEDS = list(range(500, 550))  # model selection (phase-2 choices, best run)
TEST_SEEDS = list(range(1000, 1050))  # final reported numbers only
PROGRESS_SEEDS = list(range(500, 510))  # cheap periodic evaluation for learning curves

# Settings shared by every algorithm (fairness controls)
SHARED_DEFAULTS: dict[str, Any] = {
    "total_frames": TOTAL_FRAMES,
    "seed": 0,
    "hidden": 128,
    "depth": 2,
    "gamma": 0.99,
    # Rewards are multiplied by this for learning only (returns ~ -150..+100 are
    # too large for smooth-L1 critics / Q-nets). Logged and evaluated returns
    # stay in original units. Scaling does not change the optimal policy.
    "reward_scale": 0.05,
    "eval_every": 10_000,
    "log_every": 2_000,
}


@dataclass
class AlgoSpec:
    """Describes one algorithm for the sweep runner."""

    name: str
    group: str  # "dqn" or "pg" (model sub-folder)
    defaults: dict[str, Any]
    phase1: list[tuple[str, dict[str, Any]]]  # 9 (label, overrides); first = baseline
    train_fn: Callable[[dict[str, Any], Path, str], dict[str, Any]]


# ---------------------------------------------------------------- env + nets
def obs_and_action_dims() -> tuple[int, int]:
    env = FireShieldEnv()
    return int(env.observation_space.shape[0]), int(env.action_space.n)


def make_env(seed: int, reward_scale: float = 1.0):
    """FireShield wrapped for TorchRL (identical for all algorithms).

    RewardSum sits before RewardScaling, so ``episode_reward`` is logged in
    original units while the learner sees the scaled reward.
    """
    from torchrl.envs import Compose, GymWrapper, RewardScaling, RewardSum, StepCounter, TransformedEnv

    env = GymWrapper(FireShieldEnv(), categorical_action_encoding=True)
    transforms = [StepCounter(), RewardSum()]
    if reward_scale != 1.0:
        transforms.append(RewardScaling(loc=0.0, scale=reward_scale))
    env = TransformedEnv(env, Compose(*transforms))
    env.set_seed(seed)
    return env


def make_collector(env, policy, cfg: dict[str, Any], frames_per_batch: int, **kwargs):
    from torchrl.collectors import Collector

    return Collector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=cfg["total_frames"],
        auto_register_policy_transforms=False,
        **kwargs,
    )


def build_mlp(in_dim: int, out_dim: int, hidden: int = 128, depth: int = 2) -> nn.Sequential:
    """Plain PyTorch MLP so saved weights load without TorchRL (e.g. in an API)."""
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------- evaluation
def argmax_policy(net: nn.Module) -> Callable[[np.ndarray], int]:
    """Deterministic policy: argmax over Q-values (DQN) or logits (PG)."""

    @torch.no_grad()
    def act(obs: np.ndarray) -> int:
        return int(net(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())

    return act


def evaluate(act_fn: Callable[[np.ndarray], int] | None, seeds: list[int], baseline: str | None = None) -> dict[str, Any]:
    """Run one episode per seed on the raw env and aggregate metrics.

    Either pass ``act_fn`` (obs -> action) or ``baseline`` in {"random", "greedy"}.
    """
    env = FireShieldEnv()
    returns, lengths, saved, outcomes = [], [], [], []
    for s in seeds:
        obs, info = env.reset(seed=s)
        env.action_space.seed(s)
        total, done = 0.0, False
        while not done:
            if baseline == "random":
                a = random_policy(env)
            elif baseline == "greedy":
                a = greedy_policy(env)
            else:
                a = act_fn(obs)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        returns.append(total)
        lengths.append(info["steps"])
        saved.append(info["forest_saved"])
        outcomes.append(info["outcome"])
    n = len(seeds)
    return {
        "episodes": n,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "contained_rate": outcomes.count("contained") / n,
        "village_safe_rate": 1.0 - outcomes.count("village_burnt") / n,
        "battery_fail_rate": outcomes.count("battery_depleted") / n,
        "timeout_rate": outcomes.count("timeout") / n,
        "mean_land_saved": float(np.mean(saved)),
        "mean_length": float(np.mean(lengths)),
        "returns": [round(x, 3) for x in returns],
    }


def batch_episode_stats(td) -> tuple[list[float], list[float]]:
    """Returns and lengths of episodes that finished inside a collector batch."""
    done = td["next", "done"].reshape(-1)
    if not bool(done.any()):
        return [], []
    rets = td["next", "episode_reward"].reshape(-1)[done]
    lens = td["next", "step_count"].reshape(-1)[done]
    return rets.tolist(), lens.float().tolist()


# ------------------------------------------------------------------- logging
class RunLogger:
    """Collects per-interval metrics, prints a verbose line, writes metrics.csv."""

    def __init__(self, run_dir: Path, tag: str, total_frames: int) -> None:
        self.run_dir = run_dir
        self.tag = tag
        self.total = total_frames
        self.rows: list[dict[str, Any]] = []
        self.start = time.time()
        self._ep_returns: list[float] = []
        self._ep_lengths: list[float] = []
        self._scalars: dict[str, list[float]] = {}
        run_dir.mkdir(parents=True, exist_ok=True)

    def add_episodes(self, returns: list[float], lengths: list[float]) -> None:
        self._ep_returns += returns
        self._ep_lengths += lengths

    def add(self, **scalars: float) -> None:
        for k, v in scalars.items():
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                self._scalars.setdefault(k, []).append(float(v))

    def flush(self, frames: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = time.time() - self.start
        row: dict[str, Any] = {
            "frames": frames,
            "time_s": round(elapsed, 1),
            "episodes": len(self._ep_returns),
            "train_return": float(np.mean(self._ep_returns)) if self._ep_returns else float("nan"),
            "train_length": float(np.mean(self._ep_lengths)) if self._ep_lengths else float("nan"),
        }
        for k, vals in self._scalars.items():
            row[k] = float(np.mean(vals))
        if extra:
            row.update(extra)
        self.rows.append(row)
        self._ep_returns, self._ep_lengths, self._scalars = [], [], {}
        self._write_csv()

        fps = frames / max(elapsed, 1e-6)
        parts = [f"[{self.tag}] {frames:>7}/{self.total}", f"ret {row['train_return']:7.1f}"]
        for key in ("loss", "loss_objective", "loss_critic", "entropy", "epsilon"):
            if key in row:
                parts.append(f"{key} {row[key]:.3f}")
        if "eval_return" in row:
            parts.append(f"eval {row['eval_return']:7.1f} (contained {row['eval_contained']:.0%})")
        parts.append(f"{fps:5.0f} fps")
        print(" | ".join(parts), flush=True)
        return row

    def _write_csv(self) -> None:
        keys: list[str] = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with open(self.run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


def progress_eval(net: nn.Module) -> dict[str, float]:
    res = evaluate(argmax_policy(net), PROGRESS_SEEDS)
    return {"eval_return": res["mean_return"], "eval_contained": res["contained_rate"]}


# ------------------------------------------------------------ checkpointing
def model_path(group: str, algo: str, run_name: str) -> Path:
    return MODELS_DIR / group / f"{algo}_{run_name}.pt"


def save_policy(path: Path, net: nn.Module, algo: str, config: dict[str, Any], critic: nn.Module | None = None) -> None:
    obs_dim, n_actions = obs_and_action_dims()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algo": algo,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "hidden": config["hidden"],
        "depth": config["depth"],
        "policy_state_dict": net.state_dict(),
        "config": config,
        "output": "q_values" if algo == "dqn" else "logits",
    }
    if critic is not None:
        payload["critic_state_dict"] = critic.state_dict()
    torch.save(payload, path)


def load_policy(path: str | Path) -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved policy as a plain PyTorch module (argmax gives the action)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    net = build_mlp(payload["obs_dim"], payload["n_actions"], payload["hidden"], payload["depth"])
    net.load_state_dict(payload["policy_state_dict"])
    net.eval()
    return net, payload


def finish_run(
    algo: str,
    group: str,
    run_name: str,
    run_dir: Path,
    config: dict[str, Any],
    policy_net: nn.Module,
    logger: RunLogger,
    critic: nn.Module | None = None,
) -> dict[str, Any]:
    """Final validation + test evaluation, save model and summary.json."""
    policy_net.eval()
    act = argmax_policy(policy_net)
    val = evaluate(act, VAL_SEEDS)
    test = evaluate(act, TEST_SEEDS)
    mpath = model_path(group, algo, run_name)
    save_policy(mpath, policy_net, algo, config, critic)

    tail = [r["train_return"] for r in logger.rows[-max(1, len(logger.rows) // 10) :] if not math.isnan(r["train_return"])]
    summary = {
        "algo": algo,
        "run": run_name,
        "config": config,
        "overrides": config.get("overrides", {}),
        "train_time_s": round(time.time() - logger.start, 1),
        "final_train_return": float(np.mean(tail)) if tail else None,
        "val": val,
        "test": test,
        "model_path": str(mpath.relative_to(ROOT)),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{algo}/{run_name}] DONE in {summary['train_time_s'] / 60:.1f} min | "
        f"val {val['mean_return']:.1f}±{val['std_return']:.1f} (contained {val['contained_rate']:.0%}, "
        f"village safe {val['village_safe_rate']:.0%}) | test {test['mean_return']:.1f}",
        flush=True,
    )
    return summary


def write_config(run_dir: Path, config: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

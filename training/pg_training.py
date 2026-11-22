"""Policy-gradient training on FireShield with TorchRL: REINFORCE, PPO, A2C.

Shared TorchRL components: GymWrapper env, ProbabilisticActor (Categorical
over the 8 actions), ValueOperator critic, Collector.
    REINFORCE  Monte-Carlo returns over complete episodes, optional learned
               baseline, optional return normalisation, entropy bonus.
    PPO        GAE advantages + ClipPPOLoss, several epochs of minibatches.
    A2C        GAE / n-step advantages + A2CLoss, one update per rollout.

Run directly:  uv run python -m training.pg_training --algo ppo --frames 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

from training.common import (
    LOGS_DIR,
    SHARED_DEFAULTS,
    AlgoSpec,
    RunLogger,
    batch_episode_stats,
    build_mlp,
    finish_run,
    make_collector,
    make_env,
    obs_and_action_dims,
    progress_eval,
    set_seed,
    write_config,
)


# ------------------------------------------------------------ shared builders
def build_actor_critic(env, cfg: dict[str, Any], with_critic: bool = True):
    obs_dim, n_actions = obs_and_action_dims()
    policy_net = build_mlp(obs_dim, n_actions, cfg["hidden"], cfg["depth"])
    actor = ProbabilisticActor(
        TensorDictModule(policy_net, in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"],
        spec=env.action_spec,
        distribution_class=Categorical,
        return_log_prob=True,
    )
    critic_net = critic = None
    if with_critic:
        critic_net = build_mlp(obs_dim, 1, cfg["hidden"], cfg["depth"])
        critic = ValueOperator(critic_net, in_keys=["observation"])
    return policy_net, actor, critic_net, critic


def _total_loss(out) -> torch.Tensor:
    """Sum of every loss term TorchRL returned (loss_entropy is absent when
    the entropy coefficient is 0)."""
    return sum(v for k, v in out.items() if k.startswith("loss_"))


def _entropy(out) -> float | None:
    return out["entropy"].item() if "entropy" in out.keys() else None


def _maybe_log(logger: RunLogger, cfg, frames: int, state: dict, policy_net) -> None:
    if frames >= state["next_log"] or frames >= cfg["total_frames"]:
        extra = None
        if frames >= state["next_eval"] or frames >= cfg["total_frames"]:
            extra = progress_eval(policy_net)
            state["next_eval"] += cfg["eval_every"]
        logger.flush(frames, extra)
        state["next_log"] += cfg["log_every"]


# ================================================================ REINFORCE
REINFORCE_DEFAULTS: dict[str, Any] = {
    **SHARED_DEFAULTS,
    "lr": 1e-3,
    "batch_frames": 500,  # env steps per policy update (~10 episodes)
    "normalize_returns": True,
    "baseline": False,  # learned state-value baseline
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 1.0,
}

REINFORCE_PHASE1: list[tuple[str, dict[str, Any]]] = [
    ("baseline", {}),
    ("lr_3e-4", {"lr": 3e-4}),
    ("lr_3e-3", {"lr": 3e-3}),
    ("gamma_0.95", {"gamma": 0.95}),
    ("value_baseline", {"baseline": True}),
    ("no_return_norm", {"normalize_returns": False}),
    ("entropy_0", {"entropy_coef": 0.0}),
    ("entropy_0.05", {"entropy_coef": 0.05}),
    ("batch_2000", {"batch_frames": 2000}),
]


def discounted_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float):
    """Monte-Carlo returns G_t within each episode, plus a mask that drops the
    trailing incomplete episode (whose return is unknown)."""
    r = rewards.tolist()
    d = dones.tolist()
    T = len(r)
    returns = [0.0] * T
    g = 0.0
    for t in reversed(range(T)):
        g = r[t] + (0.0 if d[t] else gamma * g)
        returns[t] = g
    last_done = max((t for t in range(T) if d[t]), default=-1)
    valid = torch.arange(T) <= last_done
    return torch.tensor(returns, dtype=torch.float32), valid


def train_reinforce(config: dict[str, Any], run_dir: Path, run_name: str) -> dict[str, Any]:
    cfg = {**REINFORCE_DEFAULTS, **config}
    write_config(run_dir, cfg)
    set_seed(cfg["seed"])
    env = make_env(cfg["seed"], cfg["reward_scale"])
    policy_net, actor, critic_net, critic = build_actor_critic(env, cfg, with_critic=cfg["baseline"])
    params = list(actor.parameters()) + (list(critic.parameters()) if critic is not None else [])
    optimizer = torch.optim.Adam(params, lr=cfg["lr"])
    # reset_at_each_iter: every batch starts a fresh episode, so returns are exact
    collector = make_collector(env, actor, cfg, cfg["batch_frames"], reset_at_each_iter=True)

    logger = RunLogger(run_dir, f"reinforce/{run_name}", cfg["total_frames"])
    frames, state = 0, {"next_log": cfg["log_every"], "next_eval": cfg["eval_every"]}
    for data in collector:
        frames += data.numel()
        logger.add_episodes(*batch_episode_stats(data))
        returns, valid = discounted_returns(
            data["next", "reward"].reshape(-1), data["next", "done"].reshape(-1), cfg["gamma"]
        )
        if valid.any():
            td = data.reshape(-1)[valid]
            g = returns[valid]
            dist = actor.get_dist(td)
            log_prob = dist.log_prob(td["action"])
            entropy = dist.entropy().mean()
            value_loss = torch.tensor(0.0)
            if critic is not None:
                values = critic(td.clone())["state_value"].squeeze(-1)
                value_loss = torch.nn.functional.mse_loss(values, g)
                adv = g - values.detach()
            else:
                adv = g
            if cfg["normalize_returns"] and adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = -(log_prob * adv).mean()
            loss = policy_loss - cfg["entropy_coef"] * entropy + cfg["value_coef"] * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            optimizer.step()
            logger.add(
                loss=loss.item(),
                loss_objective=policy_loss.item(),
                loss_critic=value_loss.item() if critic is not None else None,
                entropy=entropy.item(),
            )
        _maybe_log(logger, cfg, frames, state, policy_net)

    collector.shutdown()
    return finish_run("reinforce", "pg", run_name, run_dir, cfg, policy_net, logger, critic_net)


# ===================================================================== PPO
PPO_DEFAULTS: dict[str, Any] = {
    **SHARED_DEFAULTS,
    "lr": 3e-4,
    "gae_lambda": 0.95,
    "rollout_frames": 2048,
    "epochs": 10,
    "minibatch_size": 256,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "normalize_advantage": True,
    "max_grad_norm": 0.5,
}

PPO_PHASE1: list[tuple[str, dict[str, Any]]] = [
    ("baseline", {}),
    ("lr_1e-4", {"lr": 1e-4}),
    ("lr_1e-3", {"lr": 1e-3}),
    ("clip_0.1", {"clip_epsilon": 0.1}),
    ("clip_0.3", {"clip_epsilon": 0.3}),
    ("gae_lambda_0.8", {"gae_lambda": 0.8}),
    ("rollout_512", {"rollout_frames": 512}),
    ("entropy_0.05", {"entropy_coef": 0.05}),
    ("value_coef_1.0", {"value_coef": 1.0}),
]


def train_ppo(config: dict[str, Any], run_dir: Path, run_name: str) -> dict[str, Any]:
    cfg = {**PPO_DEFAULTS, **config}
    write_config(run_dir, cfg)
    set_seed(cfg["seed"])
    env = make_env(cfg["seed"], cfg["reward_scale"])
    policy_net, actor, critic_net, critic = build_actor_critic(env, cfg)
    advantage = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=False)
    loss_module = ClipPPOLoss(
        actor,
        critic,
        clip_epsilon=cfg["clip_epsilon"],
        entropy_coeff=cfg["entropy_coef"],
        critic_coeff=cfg["value_coef"],
        loss_critic_type="smooth_l1",
        normalize_advantage=cfg["normalize_advantage"],
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])
    collector = make_collector(env, actor, cfg, cfg["rollout_frames"])

    logger = RunLogger(run_dir, f"ppo/{run_name}", cfg["total_frames"])
    frames, state = 0, {"next_log": cfg["log_every"], "next_eval": cfg["eval_every"]}
    for data in collector:
        frames += data.numel()
        logger.add_episodes(*batch_episode_stats(data))
        with torch.no_grad():
            advantage(data)
        flat = data.reshape(-1)
        n = flat.numel()
        for _ in range(cfg["epochs"]):
            perm = torch.randperm(n)
            for start in range(0, n, cfg["minibatch_size"]):
                mb = flat[perm[start : start + cfg["minibatch_size"]]]
                out = loss_module(mb)
                loss = _total_loss(out)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg["max_grad_norm"])
                optimizer.step()
                logger.add(
                    loss=loss.item(),
                    loss_objective=out["loss_objective"].item(),
                    loss_critic=out["loss_critic"].item(),
                    entropy=_entropy(out),
                    clip_fraction=out.get("clip_fraction", torch.tensor(float("nan"))).item(),
                    kl_approx=out.get("kl_approx", torch.tensor(float("nan"))).item(),
                )
        _maybe_log(logger, cfg, frames, state, policy_net)

    collector.shutdown()
    return finish_run("ppo", "pg", run_name, run_dir, cfg, policy_net, logger, critic_net)


# ===================================================================== A2C
A2C_DEFAULTS: dict[str, Any] = {
    **SHARED_DEFAULTS,
    "lr": 7e-4,
    "gae_lambda": 0.95,
    "rollout_frames": 16,  # n-step rollout length per update
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
}

A2C_PHASE1: list[tuple[str, dict[str, Any]]] = [
    ("baseline", {}),
    ("lr_3e-4", {"lr": 3e-4}),
    ("lr_2e-3", {"lr": 2e-3}),
    ("rollout_5", {"rollout_frames": 5}),
    ("rollout_64", {"rollout_frames": 64}),
    ("gae_lambda_1.0", {"gae_lambda": 1.0}),
    ("entropy_0", {"entropy_coef": 0.0}),
    ("entropy_0.05", {"entropy_coef": 0.05}),
    ("value_coef_1.0", {"value_coef": 1.0}),
]


def train_a2c(config: dict[str, Any], run_dir: Path, run_name: str) -> dict[str, Any]:
    cfg = {**A2C_DEFAULTS, **config}
    write_config(run_dir, cfg)
    set_seed(cfg["seed"])
    env = make_env(cfg["seed"], cfg["reward_scale"])
    policy_net, actor, critic_net, critic = build_actor_critic(env, cfg)
    advantage = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=False)
    loss_module = A2CLoss(
        actor,
        critic,
        entropy_coeff=cfg["entropy_coef"],
        critic_coeff=cfg["value_coef"],
        loss_critic_type="smooth_l1",
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])
    collector = make_collector(env, actor, cfg, cfg["rollout_frames"])

    logger = RunLogger(run_dir, f"a2c/{run_name}", cfg["total_frames"])
    frames, state = 0, {"next_log": cfg["log_every"], "next_eval": cfg["eval_every"]}
    for data in collector:
        frames += data.numel()
        logger.add_episodes(*batch_episode_stats(data))
        with torch.no_grad():
            advantage(data)
        out = loss_module(data.reshape(-1))
        loss = _total_loss(out)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg["max_grad_norm"])
        optimizer.step()
        logger.add(
            loss=loss.item(),
            loss_objective=out["loss_objective"].item(),
            loss_critic=out["loss_critic"].item(),
            entropy=_entropy(out),
        )
        _maybe_log(logger, cfg, frames, state, policy_net)

    collector.shutdown()
    return finish_run("a2c", "pg", run_name, run_dir, cfg, policy_net, logger, critic_net)


SPECS = {
    "reinforce": AlgoSpec("reinforce", "pg", REINFORCE_DEFAULTS, REINFORCE_PHASE1, train_reinforce),
    "ppo": AlgoSpec("ppo", "pg", PPO_DEFAULTS, PPO_PHASE1, train_ppo),
    "a2c": AlgoSpec("a2c", "pg", A2C_DEFAULTS, A2C_PHASE1, train_a2c),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one policy-gradient run on FireShield")
    parser.add_argument("--algo", choices=list(SPECS), required=True)
    parser.add_argument("--frames", type=int, default=SHARED_DEFAULTS["total_frames"])
    parser.add_argument("--name", default="single")
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = SPECS[args.algo]
    spec.train_fn({"total_frames": args.frames}, LOGS_DIR / args.algo / args.name, args.name)

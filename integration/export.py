"""Export a trained policy to deployment formats.

    uv run main.py export                    # best model overall -> models/export/
    uv run main.py export --algo dqn

Writes:
  policy.torchscript.pt   TorchScript graph - loads with torch alone, no project code
  policy_weights.npz      raw layer weights - runs with NumPy alone (embedded / mobile / web)
  policy_card.json        input/output contract: observation layout, action names, metrics
The export is verified: all three formats must pick the same action on 200 test observations.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from environment.custom_env import ACTION_NAMES, FireShieldEnv
from integration.policy import EXPORT_DIR, ROOT, FireShieldPolicy


def _collect_observations(n: int = 200) -> np.ndarray:
    env, obs_list, rng = FireShieldEnv(), [], np.random.default_rng(0)
    seed = 1000
    obs, _ = env.reset(seed=seed)
    while len(obs_list) < n:
        obs_list.append(obs)
        obs, _, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
        if term or trunc:
            seed += 1
            obs, _ = env.reset(seed=seed)
    return np.stack(obs_list)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def export(checkpoint: Path, out_dir: Path = EXPORT_DIR) -> dict:
    import torch

    from training.common import load_policy

    out_dir.mkdir(parents=True, exist_ok=True)
    net, payload = load_policy(checkpoint)
    env = FireShieldEnv()

    # 1) TorchScript
    example = torch.zeros(1, payload["obs_dim"])
    scripted = torch.jit.trace(net, example)
    ts_path = out_dir / "policy.torchscript.pt"
    scripted.save(str(ts_path))

    # 2) NumPy weights (Linear layers in order)
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    arrays = {}
    for i, layer in enumerate(linears):
        arrays[f"W{i}"] = layer.weight.detach().numpy()
        arrays[f"b{i}"] = layer.bias.detach().numpy()
    npz_path = out_dir / "policy_weights.npz"
    np.savez(npz_path, **arrays)

    # 3) Policy card (the API contract)
    results_path = ROOT / "logs" / "results_summary.json"
    test_metrics = None
    if results_path.exists():
        res = json.loads(results_path.read_text(encoding="utf-8")).get(payload["algo"], {})
        if res:
            t = res["test"]
            test_metrics = {k: t[k] for k in ("mean_return", "contained_rate", "village_safe_rate", "mean_land_saved")}
    card = {
        "name": "FireShield wildfire response drone policy",
        "algo": payload["algo"],
        "exported": date.today().isoformat(),
        "source_checkpoint": _rel(checkpoint),
        "obs_dim": payload["obs_dim"],
        "n_actions": payload["n_actions"],
        "hidden": payload["hidden"],
        "depth": payload["depth"],
        "architecture": f"MLP {payload['obs_dim']} -> " + " -> ".join([str(payload["hidden"])] * payload["depth"])
                        + f" -> {payload['n_actions']} (ReLU)",
        "output": payload["output"],
        "decision_rule": "action = argmax(output)",
        "actions": list(ACTION_NAMES),
        "observation_layout": [
            {"name": name, "size": size, "low": float(lo), "high": float(hi)} for name, size, lo, hi in env.obs_layout
        ],
        "env_config": {"grid_size": env.n, "view_radius": env.cfg.view_radius,
                       "water_capacity": env.cfg.water_capacity, "battery_capacity": env.cfg.battery_capacity},
        "test_metrics_seeds_1000_1049": test_metrics,
        "files": {"torchscript": ts_path.name, "numpy_weights": npz_path.name},
    }
    card_path = out_dir / "policy_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")

    # verify: all formats agree
    obs = _collect_observations()
    policies = {
        "checkpoint": FireShieldPolicy.from_checkpoint(checkpoint),
        "torchscript": FireShieldPolicy.from_torchscript(ts_path),
        "numpy": FireShieldPolicy.from_numpy(npz_path),
    }
    actions = {k: [p.act(o)["action"] for o in obs] for k, p in policies.items()}
    agree = {k: float(np.mean(np.array(v) == np.array(actions["checkpoint"]))) for k, v in actions.items()}
    print(f"Exported {payload['algo'].upper()} policy from {card['source_checkpoint']} -> {_rel(out_dir)}")
    for f in (ts_path, npz_path, card_path):
        print(f"  {f.name:<24} {f.stat().st_size / 1024:7.1f} KB")
    print("  action agreement with the checkpoint on 200 test observations: "
          + ", ".join(f"{k} {v:.0%}" for k, v in agree.items()))
    if min(agree.values()) < 1.0:
        raise RuntimeError("Exported formats disagree with the checkpoint")
    return card

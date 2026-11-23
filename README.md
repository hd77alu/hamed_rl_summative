# FireShield: Wildfire Response Drone

This RL project employes a drone  environment with a limited water tank that learns to contain a spreading wildfire and protect a village. Four TorchRL agents (DQN, REINFORCE, PPO, A2C) are trained and compared on the same custom Gymnasium environment.

## Project Structure
```text
hamed_rl_summative/
├── pyproject.toml
├── uv.lock
├── README.md
├── main.py                  # CLI: play (default), train, sweep, finalize, evaluate, plots, export, serve
├── play.py                  # shortcut for `main.py play`
├── environment/
│   ├── __init__.py          # registers FireShield-v0
│   ├── custom_env.py        # Gymnasium FireShieldEnv (fire dynamics, rewards, observations)
│   ├── rendering.py         # Pygame 2.5D isometric renderer + HUD
│   └── baselines.py         # random and greedy reference policies
├── training/
│   ├── __init__.py
│   ├── common.py            # shared env factory, MLPs, evaluation, logging, checkpoints
│   ├── dqn_training.py      # TorchRL DQN + phase-1 sweep definition
│   ├── pg_training.py       # TorchRL REINFORCE, PPO, A2C + phase-1 sweep definitions
│   ├── sweep.py             # two-phase sweep, robustness runs, final selection
│   └── plots.py             # report figures + generalization evaluation
├── integration/
│   ├── policy.py            # FireShieldPolicy (checkpoint / TorchScript / NumPy) + JSON state -> obs
│   ├── export.py            # export + cross-format verification
│   ├── server.py            # JSON inference API (standard library http.server)
│   └── client_demo.py       # drone simulator controlled through the API
├── models/
│   ├── dqn/                 # dqn_<run>.pt, dqn_best.pt
│   ├── pg/                  # {reinforce,ppo,a2c}_<run>.pt, *_best.pt
│   └── export/              # policy.torchscript.pt, policy_weights.npz, policy_card.json
├── logs/                    # per-run config.json, metrics.csv, summary.json; sweep tables
├── assets/                  # plots and environment screenshots for the report
└── tests/                   # environment + integration tests (uv run --extra dev pytest)
```

## Quick Start
```bash
uv sync
uv run main.py            # play the best trained agent in the GUI (same as: uv run play.py)
```

## Training & Experiments
```bash
# Hyperparameter sweep (11 runs per algorithm, 150k steps each, runs in parallel)
uv run main.py sweep --algo all --phase 1        # runs 01-09: baseline + one change each
uv run main.py sweep --algo all --phase 2        # runs 10-11: combine top-2 / top-3 changes
uv run main.py sweep --algo all --phase robust   # best config retrained with seeds 1, 2
uv run main.py finalize                          # copy best models, write test-seed results

# Single runs / evaluation / demo
uv run main.py train --algo ppo --set lr=1e-3 clip_epsilon=0.3 --name my_run
uv run main.py evaluate                          # best models + baselines on test seeds
uv run main.py play --algo dqn --episodes 3      # GUI with verbose per-step terminal output
uv run main.py play --policy greedy --headless   # hand-coded baseline, terminal only
```


```bash
uv run main.py plots          # report figures + result tables -> assets/plots/
```

## Integration: Using the Trained Policy Outside Training
```bash
uv run main.py export         # best model -> models/export/ (TorchScript + NumPy weights + policy card)
uv run main.py serve          # JSON inference API on http://127.0.0.1:8000
uv run main.py client-demo    # simulated drone flies 3 episodes, every action comes from the API
```
- **Serialization:** the training checkpoint (`models/pg/ppo_best.pt`) is exported to `policy.torchscript.pt` (needs only PyTorch) and `policy_weights.npz` (needs only NumPy, which suits an on-board computer or a mobile/web client). `policy_card.json` describes the inputs and outputs: the observation layout, action names and test metrics. During export, all three formats must choose the same action on 200 test observations.
- **API:** `GET /health`, `GET /spec`, `GET /example`, `POST /act`. A drone posts its world state as JSON and gets back an action:
  ```json
  POST /act  {"state": {"drone": [7, 4], "base": [7, 4], "village_cells": [[10, 0], ...], "water": 4,
                        "battery": 80, "wind": {"direction": "N", "speed": 0.82}, "dryness": 0.95,
                        "fire_intensity": [[...12x12...]], "fuel": [[...12x12...]], "burnt": [[...]]}}
  -> {"action": 8, "action_name": "fly_to_fire", "confidence": 0.979, "probabilities": {...}, "latency_ms": 1.8}
  ```
  The API converts the state to an observation with the environment's own code, so a real controller and the simulator always see identical inputs. Requests can also send the raw 65-value `observation`. Invalid input returns HTTP 422 with an error message. The server uses only the Python standard library, so it adds no dependencies.
- **End-to-end check:** `client-demo` runs whole episodes over HTTP. Measured: 103 decisions with 0 mismatches against the local checkpoint and a median round trip of ~4 ms. The outcomes match `main.py play` exactly.


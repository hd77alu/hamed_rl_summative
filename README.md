# FireShield: Wildfire Response Drone

This RL project employes a drone  environment with a limited water tank that learns to contain a spreading wildfire and protect a village. Four TorchRL agents (DQN, REINFORCE, PPO, A2C) are trained and compared on the same custom Gymnasium environment.

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
Options: `--workers N` (default: CPU cores − 2), `--frames N` (smoke tests), `--force` (rerun finished runs). Finished runs are skipped, so an interrupted sweep can simply be restarted.

Runs are selected on **validation seeds 500–549**. Final numbers are reported on held-out **test seeds 1000–1049**. The full protocol is in [assignment.md](assignment.md) §5.

## Project Structure
```text
hamed_rl_summative/
├── pyproject.toml
├── uv.lock
├── README.md
├── main.py                  # CLI: play (default), train, sweep, finalize, evaluate
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
│   └── sweep.py             # two-phase sweep, robustness runs, final selection
├── models/
│   ├── dqn/                 # dqn_<run>.pt, dqn_best.pt
│   └── pg/                  # {reinforce,ppo,a2c}_<run>.pt, *_best.pt
├── logs/                    # per-run config.json, metrics.csv, summary.json; sweep tables
├── assets/                  # plots and environment screenshots for the report
└── tests/                   # environment unit tests (uv run --extra dev pytest)
```

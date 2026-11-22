# FireShield: Wildfire Response Drone

This RL project employes a drone  environment with a limited water tank that learns to contain a spreading wildfire and protect a village. Four TorchRL agents (DQN, REINFORCE, PPO, A2C) are trained and compared on the same custom Gymnasium environment.

## Quick Start
```bash
uv sync
uv run main.py
```

## Project Structure
```text
hamed_rl_summative/
├── pyproject.toml
├── uv.lock
├── README.md
├── main.py                  # Entry point to train, evaluate, or play best model
├── environment/
│   ├── __init__.py
│   ├── custom_env.py        # Gymnasium-compliant WildfireEnv
│   └── rendering.py         # Pygame / OpenGL 2.5D isometric simulation
├── training/
│   ├── __init__.py
│   ├── dqn_training.py      # TorchRL DQN training + hyperparameter sweep
│   └── pg_training.py       # TorchRL REINFORCE, PPO, and A2C training + sweeps
├── models/
│   ├── dqn/
│   └── pg/
├── logs/                    # CSV/JSON metrics for rewards, loss, entropy, convergence
├── assets/                  # Saved plots and environment screenshots for report
└── tests/                   # Unit tests verifying action/obs shapes & determinism
```
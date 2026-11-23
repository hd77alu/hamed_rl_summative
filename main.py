"""FireShield command-line entry point.

    uv run main.py                                  play the best trained agent (GUI)
    uv run main.py play --algo ppo --episodes 3     play a specific algorithm
    uv run main.py train --algo dqn --set lr=1e-3   one training run
    uv run main.py sweep --algo all --phase 1       hyperparameter sweep (1 | 2 | robust)
    uv run main.py finalize                         select best models, test-seed results
    uv run main.py evaluate                         evaluate best models + baselines
    uv run main.py plots                            report figures -> assets/plots/
    uv run main.py export                           best model -> TorchScript + NumPy + policy card
    uv run main.py serve --port 8000                JSON inference API (POST /act)
    uv run main.py client-demo                      drone simulator driven through the API
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALGOS = ("dqn", "reinforce", "ppo", "a2c")


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = {"True": True, "False": False}.get(raw, raw)
    return out


def best_model_path(algo: str | None) -> Path | None:
    """Model to play: the requested algorithm, else the best on the test seeds."""
    group = lambda a: "dqn" if a == "dqn" else "pg"  # noqa: E731
    if algo:
        p = ROOT / "models" / group(algo) / f"{algo}_best.pt"
        return p if p.exists() else None
    summary = ROOT / "logs" / "results_summary.json"
    if summary.exists():
        res = json.loads(summary.read_text(encoding="utf-8"))
        cands = [(res[a]["test"]["mean_return"], a) for a in ALGOS if a in res]
        if cands:
            return ROOT / res[max(cands)[1]]["model"]
    for a in ("ppo", "a2c", "dqn", "reinforce"):
        p = ROOT / "models" / group(a) / f"{a}_best.pt"
        if p.exists():
            return p
    return None


# ------------------------------------------------------------------ commands
def cmd_play(args) -> None:
    from environment import ACTION_NAMES, FireShieldEnv
    from environment.baselines import greedy_policy, random_policy

    if args.policy in ("greedy", "random"):
        label, act = args.policy, None
    else:
        path = Path(args.model) if args.model else best_model_path(args.algo)
        if path is None or not path.exists():
            print("No trained model found (run training + `uv run main.py finalize` first). Showing the greedy baseline.")
            label, act = "greedy", None
            args.policy = "greedy"
        else:
            from training.common import argmax_policy, load_policy

            net, meta = load_policy(path)
            label, act = f"{meta['algo'].upper()} ({path.relative_to(ROOT) if path.is_relative_to(ROOT) else path})", argmax_policy(net)

    env = FireShieldEnv(render_mode=None if args.headless else "human")
    print(f"\nFireShield | policy: {label} | episodes: {args.episodes} | start seed: {args.seed}\n")
    for ep in range(args.episodes):
        seed = args.seed + ep
        obs, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        done, total = False, 0.0
        print(f"--- Episode {ep + 1} (seed {seed}) | village at {tuple(env.village_cells[0])}, base at {env.base_pos} | "
              f"wind toward {['N','NE','E','SE','S','SW','W','NW'][env.wind_dir]} @ {env.wind_speed:.2f}")
        while not done:
            if args.policy == "greedy":
                a = greedy_policy(env)
            elif args.policy == "random":
                a = random_policy(env)
            else:
                a = act(obs)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
            if args.verbose:
                print(f"step {info['steps']:>3} | {ACTION_NAMES[a]:<16} | r {r:+7.2f} | return {total:+8.2f} | "
                      f"water {info['water']} | battery {info['battery']:>2} | fires {info['fires_active']:>2} | "
                      f"land saved {info['forest_saved']:.0%}")
        print(f"=== Episode {ep + 1} result: {info['outcome'].upper()} | return {total:.1f} | steps {info['steps']} | "
              f"land saved {info['forest_saved']:.0%} | extinguished {info['cells_extinguished']} | "
              f"water used {info['water_used']} | firebreaks {info['firebreaks_built']}\n")
        if not args.headless:
            time.sleep(1.5)
    env.close()


def cmd_train(args) -> None:
    import torch

    from training.common import LOGS_DIR
    from training.sweep import get_spec

    torch.set_num_threads(args.threads)
    overrides = parse_overrides(args.set)
    config = {**overrides, "overrides": overrides, "phase": "manual", "seed": args.seed}
    if args.frames:
        config["total_frames"] = args.frames
    spec = get_spec(args.algo)
    spec.train_fn(config, LOGS_DIR / args.algo / args.name, args.name)


def cmd_sweep(args) -> None:
    from training.sweep import run_phase

    run_phase(args.algo, args.phase, args.workers, args.frames, args.force)


def cmd_finalize(args) -> None:
    from training.sweep import finalize

    finalize()


def cmd_evaluate(args) -> None:
    from training.common import TEST_SEEDS, argmax_policy, evaluate, load_policy

    rows = []
    for algo in ALGOS:
        p = best_model_path(algo)
        if p is None:
            continue
        net, _ = load_policy(p)
        rows.append((algo, evaluate(argmax_policy(net), TEST_SEEDS)))
    for name in ("greedy", "random"):
        rows.append((f"baseline_{name}", evaluate(None, TEST_SEEDS, baseline=name)))
    print(f"\nTest seeds 1000-1049\n{'policy':<18}{'return':>16}{'contained':>11}{'village safe':>14}{'land saved':>12}{'length':>8}")
    for name, r in rows:
        print(f"{name:<18}{r['mean_return']:>9.1f}±{r['std_return']:<6.1f}{r['contained_rate']:>10.0%}"
              f"{r['village_safe_rate']:>14.0%}{r['mean_land_saved']:>12.0%}{r['mean_length']:>8.1f}")


def cmd_plots(args) -> None:
    from training.plots import make_all

    make_all(regen=args.regen)


def cmd_export(args) -> None:
    from integration.export import export

    path = best_model_path(args.algo)
    if path is None:
        sys.exit("No trained model found - run training and `uv run main.py finalize` first.")
    export(path)


def cmd_serve(args) -> None:
    from integration.server import serve

    serve(args.host, args.port, args.backend, args.algo)


def cmd_client_demo(args) -> None:
    from integration.client_demo import run

    run(args.url, args.episodes, args.seed, args.backend)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FireShield: wildfire response drone RL")
    sub = p.add_subparsers(dest="command")

    play = sub.add_parser("play", help="run an agent in the GUI")
    play.add_argument("--algo", choices=ALGOS)
    play.add_argument("--model", help="path to a .pt checkpoint")
    play.add_argument("--policy", choices=("model", "greedy", "random"), default="model")
    play.add_argument("--episodes", type=int, default=3)
    play.add_argument("--seed", type=int, default=1000, help="first episode seed (1000+ = unseen test seeds)")
    play.add_argument("--headless", action="store_true", help="no window, terminal output only")
    play.add_argument("--quiet", dest="verbose", action="store_false", help="no per-step output")
    play.set_defaults(func=cmd_play)

    train = sub.add_parser("train", help="single training run")
    train.add_argument("--algo", choices=ALGOS, required=True)
    train.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="hyperparameter overrides")
    train.add_argument("--frames", type=int)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--name", default="manual")
    train.add_argument("--threads", type=int, default=4)
    train.set_defaults(func=cmd_train)

    sweep = sub.add_parser("sweep", help="hyperparameter sweep phase")
    sweep.add_argument("--algo", choices=(*ALGOS, "all"), default="all")
    sweep.add_argument("--phase", choices=("1", "2", "robust"), required=True)
    sweep.add_argument("--workers", type=int, help="parallel processes (default: CPU cores - 2)")
    sweep.add_argument("--frames", type=int, help="override the 150k budget (smoke tests only)")
    sweep.add_argument("--force", action="store_true", help="rerun finished runs")
    sweep.set_defaults(func=cmd_sweep)

    fin = sub.add_parser("finalize", help="select best model per algorithm, write results")
    fin.set_defaults(func=cmd_finalize)

    ev = sub.add_parser("evaluate", help="evaluate best models and baselines on test seeds")
    ev.set_defaults(func=cmd_evaluate)

    pl = sub.add_parser("plots", help="generate report figures from the logs")
    pl.add_argument("--regen", action="store_true", help="recompute the cached generalization evaluation")
    pl.set_defaults(func=cmd_plots)

    ex = sub.add_parser("export", help="export the best model to TorchScript + NumPy + policy card")
    ex.add_argument("--algo", choices=ALGOS, help="default: best algorithm on the test seeds")
    ex.set_defaults(func=cmd_export)

    sv = sub.add_parser("serve", help="run the JSON inference API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--backend", choices=("numpy", "torchscript", "checkpoint"), default="numpy")
    sv.add_argument("--algo", choices=ALGOS, help="checkpoint backend only")
    sv.set_defaults(func=cmd_serve)

    cd = sub.add_parser("client-demo", help="fly episodes where every action comes from the API")
    cd.add_argument("--url", help="existing server (default: start one locally)")
    cd.add_argument("--episodes", type=int, default=3)
    cd.add_argument("--seed", type=int, default=1000)
    cd.add_argument("--backend", choices=("numpy", "torchscript", "checkpoint"), default="numpy")
    cd.set_defaults(func=cmd_client_demo)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:  # `uv run main.py` -> demo the best agent
        args = parser.parse_args(["play", *(argv or [])])
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])

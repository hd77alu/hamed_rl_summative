"""Report figures for FireShield, built from the training logs.

    uv run main.py plots                 # all figures -> assets/plots/
    uv run main.py plots --regen         # also recompute the generalization evaluation

Inputs: logs/<algo>/<run>/{metrics.csv,summary.json}, logs/<algo>/robust_selection.json,
logs/results_summary.json (written by `finalize`). The generalization test (new env
conditions) is computed once and cached in logs/generalization.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.common import LOGS_DIR, ROOT

OUT_DIR = ROOT / "assets" / "plots"
ALGOS = ("dqn", "reinforce", "ppo", "a2c")
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
ROBUST = ("robust_seed1", "robust_seed2")

# Categorical slots validated all-pairs (light surface): no hard fail, CVD 6.9 in the
# warn band -> every series also gets a marker + direct label (secondary encoding).
COLOR = {"dqn": "#2a78d6", "reinforce": "#e34948", "ppo": "#1baf7a", "a2c": "#eda100"}
MARKER = {"dqn": "o", "reinforce": "s", "ppo": "D", "a2c": "^"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
GREEDY_C, RANDOM_C = "#52514e", "#a9a8a2"
STATUS = {"contained": "#0ca30c", "battery_depleted": "#fab219", "village_burnt": "#d03b3b", "timeout": "#b9b8b2"}
OUTCOME_LABEL = {"contained": "Contained", "village_burnt": "Village burnt",
                 "battery_depleted": "Battery depleted", "timeout": "Timeout"}
UNIFORM_ENTROPY = math.log(9)

GEN_CONDITIONS = {
    "Training\nconditions": {},
    "Faster spread\n(p = 0.13)": {"spread_prob": 0.13},
    "More ignitions\n(up to 5)": {"max_ignitions": 5},
    "Stronger wind\n(2.5)": {"wind_strength": 2.5},
}


# ----------------------------------------------------------------- style
def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9.5, "axes.labelcolor": INK2,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def _kframes(ax) -> None:
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.set_xlabel("Environment steps")


def _ref_line(ax, y: float, text: str, color: str, style: str, horizontal: bool = True) -> None:
    if horizontal:
        ax.axhline(y, color=color, lw=1.3, ls=style, zorder=1)
        ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=8.5, color=INK2)
    else:
        ax.axvline(y, color=color, lw=1.3, ls=style, zorder=1)


def _end_labels(ax, items: list[tuple[str, float, str]], min_gap: float) -> None:
    """Direct labels at the right edge, nudged apart so they never collide."""
    items = sorted(items, key=lambda t: t[1])
    ys: list[float] = []
    for _, y, _ in items:
        ys.append(max(y, ys[-1] + min_gap) if ys else y)
    for (text, _, color), y in zip(items, ys):
        ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, fontweight="bold", color=INK)
        ax.plot([1.0], [y], marker="s", ms=6, color=color, transform=ax.get_yaxis_transform(), clip_on=False)


def _save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


# ----------------------------------------------------------------- data
def _metrics(algo: str, run: str) -> pd.DataFrame:
    return pd.read_csv(LOGS_DIR / algo / run / "metrics.csv")


def _summary(algo: str, run: str) -> dict:
    return json.loads((LOGS_DIR / algo / run / "summary.json").read_text(encoding="utf-8"))


def _best_run(algo: str) -> str:
    return json.loads((LOGS_DIR / algo / "robust_selection.json").read_text(encoding="utf-8"))["best_run"]


def _seed_runs(algo: str) -> list[str]:
    return [_best_run(algo), *[r for r in ROBUST if (LOGS_DIR / algo / r / "summary.json").exists()]]


def _sweep_runs(algo: str) -> list[str]:
    return sorted(p.name for p in (LOGS_DIR / algo).glob("run_*") if (p / "summary.json").exists())


def _results() -> dict:
    return json.loads((LOGS_DIR / "results_summary.json").read_text(encoding="utf-8"))


def _seed_curves(algo: str, column: str, smooth: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Frames and a (n_seeds, n_points) array of `column` for the selected config."""
    frames, rows = None, []
    for run in _seed_runs(algo):
        df = _metrics(algo, run)[["frames", column]].dropna()
        y = df[column].rolling(smooth, min_periods=1).mean().to_numpy() if smooth > 1 else df[column].to_numpy()
        rows.append(y)
        frames = df["frames"].to_numpy() if frames is None or len(df) < len(frames) else frames
    n = min(len(r) for r in rows)
    return frames[:n], np.vstack([r[:n] for r in rows])


def _baselines() -> dict:
    res = _results()
    return {"greedy": res["baseline_greedy"]["test"], "random": res["baseline_random"]["test"]}


# ----------------------------------------------------------------- figures
def fig_comparative_rewards() -> Path:
    base = _baselines()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    panels = (("train_return", 5, "Training episode return",
               "Episodes finished while training (exploration on), smoothed"),
              ("eval_return", 1, "Greedy-policy evaluation return",
               "Deterministic policy, 10 held-out seeds, every 10k steps"))
    for ax, (col, smooth, title, sub) in zip(axes, panels):
        ends = []
        for algo in ALGOS:
            f, Y = _seed_curves(algo, col, smooth)
            m = Y.mean(0)
            ax.fill_between(f, Y.min(0), Y.max(0), color=COLOR[algo], alpha=0.14, lw=0)
            ax.plot(f, m, color=COLOR[algo], marker=MARKER[algo], markevery=max(1, len(f) // 8), ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABEL[algo])
            ends.append((LABEL[algo], m[-1], COLOR[algo]))
        ax.axhline(base["greedy"]["mean_return"], color=GREEDY_C, lw=1.3, ls="--", zorder=1)
        ax.axhline(base["random"]["mean_return"], color=RANDOM_C, lw=1.3, ls=":", zorder=1)
        ax.text(f[0], base["greedy"]["mean_return"] + 4, "greedy heuristic (test)", fontsize=8.5, color=INK2, bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.9})
        ax.text(f[0], base["random"]["mean_return"] - 6, "random policy (test)", fontsize=8.5, color=INK2, va="top", zorder=5, bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.9})
        _end_labels(ax, ends, min_gap=9)
        ax.set_title(title)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK2)
        _kframes(ax)
    axes[0].set_ylabel("Episode return (unscaled)")
    axes[0].set_title(panels[0][2], pad=18)
    axes[1].set_title(panels[1][2], pad=18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Cumulative reward per algorithm - selected configuration, mean of 3 training seeds (band = min-max)",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", y=1.04)
    fig.subplots_adjust(wspace=0.22)
    return _save(fig, "fig1_comparative_rewards.png")


def fig_dqn_objective() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    specs = (("loss", 5, "TD loss (smooth L1)", "Loss"), ("q_mean", 1, "Mean predicted Q-value", "Q (scaled reward units)"),
             ("epsilon", 1, "Exploration rate (epsilon)", "Epsilon"))
    runs = _seed_runs("dqn")
    for ax, (col, smooth, title, ylab) in zip(axes, specs):
        f, Y = _seed_curves("dqn", col, smooth)
        for i, y in enumerate(Y):
            ax.plot(f, y, color=COLOR["dqn"], lw=1, alpha=0.35, label="individual seeds" if i == 0 else None)
        ax.plot(f, Y.mean(0), color=COLOR["dqn"], lw=2.2, label="mean of 3 seeds")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        _kframes(ax)
    axes[0].legend(loc="upper right")
    best = _summary("dqn", runs[0])
    fig.suptitle(f"DQN objective - selected configuration {runs[0]} (lr = {best['config']['lr']:g}), 3 seeds",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", y=1.05)
    fig.subplots_adjust(wspace=0.3)
    return _save(fig, "fig2_dqn_objective.png")


def fig_pg_entropy() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    ax = axes[0]
    ends = []
    for algo in ("reinforce", "ppo", "a2c"):
        if "entropy" in _metrics(algo, _best_run(algo)).columns:
            f, Y = _seed_curves(algo, "entropy", 3)
            ax.fill_between(f, Y.min(0), Y.max(0), color=COLOR[algo], alpha=0.14, lw=0)
            ax.plot(f, Y.mean(0), color=COLOR[algo], marker=MARKER[algo], markevery=max(1, len(f) // 8), ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"{LABEL[algo]} ({_best_run(algo)[7:]})")
            end = Y.mean(0)[-1]
        else:  # entropy_coef = 0 -> TorchRL's loss does not report entropy; show the default run instead
            df = _metrics(algo, "run_01_baseline").dropna(subset=["entropy"])
            y = df["entropy"].rolling(3, min_periods=1).mean()
            ax.plot(df["frames"], y, color=COLOR[algo], ls="--", marker=MARKER[algo], markevery=max(1, len(df) // 8),
                    ms=5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=f"{LABEL[algo]} (default run, seed 0; selected entropy-0 config logs no entropy)")
            end = y.iloc[-1]
        ends.append((LABEL[algo], end, COLOR[algo]))
    _end_labels(ax, ends, min_gap=0.09)
    ax.set_title("Selected configurations (3-seed mean, band = min-max)", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=8.5)

    ax = axes[1]
    ppo_runs = (("run_01_baseline", "baseline (clip 0.2, rollout 2048)", RANDOM_C, "-"),
                ("run_05_clip_0.3", "clip 0.3 - selected", COLOR["ppo"], "-"),
                ("run_10_combo_top2", "clip 0.3 + rollout 512", INK2, "-"),
                ("run_11_combo_top3", "+ value coef 1.0", INK2, "--"))
    ends = []
    for run, lab, color, ls in ppo_runs:
        df = _metrics("ppo", run).dropna(subset=["entropy"])
        y = df["entropy"].rolling(3, min_periods=1).mean()
        ret = _summary("ppo", run)["val"]["mean_return"]
        ax.plot(df["frames"], y, color=color, ls=ls, lw=2.2 if "selected" in lab else 1.8,
                label=f"{lab}  (val return {ret:.0f})")
        ends.append((run[4:6], y.iloc[-1], color))
    ax.set_title("PPO: combining top settings collapses entropy early", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=8.5)
    for a in axes:
        a.axhline(UNIFORM_ENTROPY, color=MUTED, lw=1.3, ls=":", zorder=1)
        a.text(0.35, UNIFORM_ENTROPY + 0.04, "uniform random policy (ln 9)", transform=a.get_yaxis_transform(),
               fontsize=8.5, color=INK2)
        _kframes(a)
    axes[0].set_ylabel("Policy entropy (nats)")
    axes[0].set_ylim(0, 2.45)
    fig.suptitle("Policy entropy of the policy-gradient methods", x=0.06, ha="left", fontsize=12,
                 fontweight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.3)
    return _save(fig, "fig3_pg_entropy.png")


def fig_convergence() -> Path:
    base = _baselines()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True, sharey=True)
    for ax, algo in zip(axes.flat, ALGOS):
        best = _best_run(algo)
        for run in _sweep_runs(algo):
            df = _metrics(algo, run).dropna(subset=["eval_return"])
            ax.plot(df["frames"], df["eval_return"], color="#c9c8c2", lw=1, zorder=2)
        f, Y = _seed_curves(algo, "eval_return")
        for y in Y:
            ax.plot(f, y, color=COLOR[algo], lw=1, alpha=0.45, zorder=3)
        ax.plot(f, Y.mean(0), color=COLOR[algo], lw=2.4, marker=MARKER[algo], ms=5, markevery=2,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
        ax.axhline(base["greedy"]["mean_return"], color=GREEDY_C, lw=1.2, ls="--", zorder=1)
        ax.axhline(base["random"]["mean_return"], color=RANDOM_C, lw=1.2, ls=":", zorder=1)
        # frames at which the 3-seed mean first gets 50 points above the random policy
        thr = base["random"]["mean_return"] + 50
        hit = np.nonzero(Y.mean(0) >= thr)[0]
        note = f"first beats random + 50 at {f[hit[0]] / 1000:.0f}k steps" if len(hit) else "never beats random + 50"
        ax.set_title(f"{LABEL[algo]}  -  selected: {best[7:]}")
        ax.text(0.01, 0.97, note, transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
        ax.set_ylim(-172, 88)
    for ax in axes[1]:
        _kframes(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Evaluation return")
    handles = [plt.Line2D([], [], color="#c9c8c2", lw=1.2, label="all 11 sweep runs (seed 0)"),
               plt.Line2D([], [], color=INK2, lw=1, alpha=0.5, label="selected config, each seed"),
               plt.Line2D([], [], color=INK2, lw=2.4, label="selected config, 3-seed mean"),
               plt.Line2D([], [], color=GREEDY_C, lw=1.2, ls="--", label="greedy heuristic"),
               plt.Line2D([], [], color=RANDOM_C, lw=1.2, ls=":", label="random policy")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Convergence: greedy-policy evaluation return during training (10 progress seeds)",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", y=0.99)
    fig.subplots_adjust(hspace=0.28, wspace=0.08, bottom=0.11)
    return _save(fig, "fig4_convergence.png")


def fig_sweep() -> Path:
    gen = _generalization()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True)
    for ax, algo in zip(axes.flat, ALGOS):
        runs = _sweep_runs(algo)
        best = _best_run(algo)
        vals = [_summary(algo, r)["val"] for r in runs]
        y = np.arange(len(runs))[::-1]
        for yi, run, v in zip(y, runs, vals):
            se = v["std_return"] / math.sqrt(v["episodes"])
            c = COLOR[algo] if run != "run_01_baseline" else INK2
            ax.errorbar(v["mean_return"], yi, xerr=se, fmt="none", ecolor=c, elinewidth=1.4, capsize=0, zorder=2)
            ax.plot(v["mean_return"], yi, marker=MARKER[algo], ms=8 if run == best else 6.5, color=c,
                    markeredgecolor=INK if run == best else SURFACE, markeredgewidth=1.4, zorder=3)
            ax.annotate(f"{v['mean_return']:.0f}", (v["mean_return"] + se, yi), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y, [r.replace("run_", "").replace("_", " ", 1) + ("  *" if r == best else "") for r in runs],
                      fontsize=8.5)
        ax.axhline(y[8] - 0.5, color=MUTED, lw=0.8)  # phase 1 | phase 2 separator
        _ref_line(ax, gen["val_baselines"]["greedy"]["mean_return"], "", GREEDY_C, "--", horizontal=False)
        _ref_line(ax, gen["val_baselines"]["random"]["mean_return"], "", RANDOM_C, ":", horizontal=False)
        ax.set_title(LABEL[algo])
        ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Mean return on validation seeds 500-549 (whisker = 1 standard error)")
    handles = [plt.Line2D([], [], color=INK2, marker="o", lw=0, label="run 01 = default hyperparameters"),
               plt.Line2D([], [], color=MUTED, marker="o", lw=0, markeredgecolor=INK, label="* = selected (best validation)"),
               plt.Line2D([], [], color=GREEDY_C, lw=1.2, ls="--", label="greedy heuristic"),
               plt.Line2D([], [], color=RANDOM_C, lw=1.2, ls=":", label="random policy")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Hyperparameter sweep: runs 02-09 change one setting; runs 10-11 combine the top 2 / top 3 (below line)",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", y=0.99)
    fig.subplots_adjust(hspace=0.2, wspace=0.42, bottom=0.1)
    return _save(fig, "fig5_hyperparameter_sweep.png")


def fig_test_results() -> Path:
    res = _results()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    order = sorted(ALGOS, key=lambda a: np.mean(_seed_test(a)))
    for i, algo in enumerate(order):
        vals = _seed_test(algo)
        m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
        ax.errorbar(m, i, xerr=s, fmt="none", ecolor=COLOR[algo], elinewidth=2, capsize=0, zorder=2)
        ax.scatter(vals, [i] * len(vals), s=22, color=SURFACE, edgecolor=COLOR[algo], lw=1.3, zorder=3)
        ax.plot(m, i, marker=MARKER[algo], ms=9, color=COLOR[algo], markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
        ax.annotate(f"{m:.1f} ± {s:.1f}", (m, i), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK)
    ax.set_yticks(range(len(order)), [LABEL[a] for a in order])
    ax.axvline(res["baseline_greedy"]["test"]["mean_return"], color=GREEDY_C, ls="--", lw=1.3)
    ax.axvline(res["baseline_random"]["test"]["mean_return"], color=RANDOM_C, ls=":", lw=1.3)
    ax.text(res["baseline_greedy"]["test"]["mean_return"], len(order) - 0.45, " greedy", fontsize=8.5, color=INK2)
    ax.text(res["baseline_random"]["test"]["mean_return"], len(order) - 0.45, " random", fontsize=8.5, color=INK2)
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_xlabel("Mean test return (dot = mean of 3 seeds, whisker = sample std, rings = each seed)")
    ax.set_title("Test return across 3 training seeds")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    rows = [(LABEL[a], res[a]["test"]) for a in order[::-1]] + [("Greedy", res["baseline_greedy"]["test"]),
                                                               ("Random", res["baseline_random"]["test"])]
    keys = ("contained", "battery_depleted", "village_burnt", "timeout")
    rates = {"contained": "contained_rate", "battery_depleted": "battery_fail_rate", "timeout": "timeout_rate"}
    for i, (name, t) in enumerate(rows[::-1]):
        shares = {k: t[rates[k]] for k in rates}
        shares["village_burnt"] = 1 - t["village_safe_rate"]
        left = 0.0
        for k in keys:
            w = shares[k]
            if w <= 0:
                continue
            ax.barh(i, w, left=left, color=STATUS[k], height=0.62, edgecolor=SURFACE, linewidth=2)
            if w >= 0.08:
                ax.text(left + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=8.5,
                        color=INK if k != "village_burnt" else "white")
            left += w
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of 50 test episodes (selected model)")
    ax.set_title("How episodes end")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[matplotlib.patches.Patch(color=STATUS[k], label=OUTCOME_LABEL[k]) for k in keys],
              loc="upper center", ncol=4, bbox_to_anchor=(0.45, -0.14))
    fig.suptitle("Final evaluation on unseen test seeds 1000-1049", x=0.06, ha="left", fontsize=12,
                 fontweight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.28)
    return _save(fig, "fig6_test_results.png")


def fig_generalization() -> Path:
    gen = _generalization()
    conds = list(GEN_CONDITIONS)
    x = np.arange(len(conds))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax, key, ylab, fmt in ((axes[0], "mean_return", "Mean test return", "{:.0f}"),
                               (axes[1], "contained_rate", "Fires contained", "{:.0%}")):
        ends = []
        for name, color, ls, mk in (("greedy", GREEDY_C, "--", "x"), ("random", RANDOM_C, ":", "x")):
            ys = [gen["results"][c][name][key] for c in conds]
            ax.plot(x, ys, color=color, ls=ls, lw=1.5, marker=mk, ms=6)
            ends.append((name, ys[-1], color))
        for algo in ALGOS:
            ys = [gen["results"][c][algo][key] for c in conds]
            ax.plot(x, ys, color=COLOR[algo], marker=MARKER[algo], ms=7, markeredgecolor=SURFACE,
                    markeredgewidth=1.3, label=LABEL[algo])
            ends.append((LABEL[algo], ys[-1], COLOR[algo]))
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        _end_labels(ax, ends, min_gap=span * 0.065)
        ax.set_xticks(x, conds, fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_title("Return under changed fire conditions")
    axes[1].set_title("Containment rate under changed fire conditions")
    fig.suptitle("Generalization: selected models on unseen test seeds and changed, never-trained conditions",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.32)
    return _save(fig, "fig7_generalization.png")


# ------------------------------------------------- episode-based figures (report)
CONVERGENCE_WINDOW = 10  # log points (about 20k steps)
CONVERGENCE_TOL = 15.0  # return points around the final level


def _episode_curve(algo: str, run: str, smooth: int) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative finished episodes and (smoothed) training return for one run."""
    df = _metrics(algo, run)
    y = df["train_return"].rolling(smooth, min_periods=smooth).mean().to_numpy()
    return df["episodes"].cumsum().to_numpy(), y


def convergence_table() -> pd.DataFrame:
    """Per seed: episodes until training return first beats random + 50, and until it stays
    within CONVERGENCE_TOL of its final level (mean of the last CONVERGENCE_WINDOW logs)."""
    threshold = _baselines()["random"]["mean_return"] + 50
    rows = []
    for algo in ALGOS:
        for seed, run in enumerate(_seed_runs(algo)):
            ep, y = _episode_curve(algo, run, CONVERGENCE_WINDOW)
            final = float(_metrics(algo, run)["train_return"].tail(CONVERGENCE_WINDOW).mean())
            ok = np.abs(y - final) <= CONVERGENCE_TOL
            i = len(y)
            while i > 0 and ok[i - 1]:
                i -= 1
            hit = np.nonzero(y >= threshold)[0]
            rows.append({"algo": algo, "seed": seed, "total_episodes": int(ep[-1]), "final_return": round(final, 1),
                         "stable_episode": int(ep[i]), "threshold_episode": int(ep[hit[0]]) if len(hit) else None})
    return pd.DataFrame(rows)


def fig_rewards_by_episode() -> Path:
    base = _baselines()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey=True)
    for ax, algo in zip(axes.flat, ALGOS):
        for seed, run in enumerate(_seed_runs(algo)):
            ep, y = _episode_curve(algo, run, 5)
            ax.plot(ep, y, color=COLOR[algo], ls=("-", "--", ":")[seed], lw=1.8, label=f"seed {seed}")
        ax.axhline(base["greedy"]["mean_return"], color=GREEDY_C, lw=1.2, ls="--", zorder=1)
        ax.axhline(base["random"]["mean_return"], color=RANDOM_C, lw=1.2, ls=":", zorder=1)
        ax.text(0.99, base["greedy"]["mean_return"] - 4, "greedy (test)", transform=ax.get_yaxis_transform(),
                ha="right", va="top", fontsize=8.5, color=INK2)
        ax.text(0.99, base["random"]["mean_return"] - 4, "random (test)", transform=ax.get_yaxis_transform(),
                ha="right", va="top", fontsize=8.5, color=INK2)
        ax.set_title(f"{LABEL[algo]} ({_best_run(algo)[7:]})")
        ax.set_ylim(-150, 65)
        ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 0.9))
    for ax in axes[1]:
        ax.set_xlabel("Training episodes")
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode return")
    fig.subplots_adjust(hspace=0.24, wspace=0.06)
    return _save(fig, "fig8_rewards_by_episode.png")


def fig_episodes_to_converge() -> Path:
    t = convergence_table()
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.9), sharey=True)
    panels = (("threshold_episode", "First beats random by 50 points"), ("stable_episode", "Return settles at its final level"))
    for ax, (col, title) in zip(axes, panels):
        for i, algo in enumerate(ALGOS[::-1]):
            sub = t[t["algo"] == algo]
            for _, r in sub.iterrows():
                if pd.isna(r[col]):
                    continue
                learned = r["final_return"] > -64
                ax.plot(r[col], i, marker=MARKER[algo], ms=9, ls="", color=COLOR[algo] if learned else SURFACE,
                        markeredgecolor=COLOR[algo], markeredgewidth=1.6)
            missed = int(sub[col].isna().sum())
            if missed:
                ax.text(2960, i, f"{missed} of 3 seeds never", va="center", ha="right", fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(ALGOS)), [LABEL[a] for a in ALGOS[::-1]])
        ax.set_ylim(-0.6, 3.6)
        ax.set_xlim(0, 3000)
        ax.set_xlabel("Training episodes (filled = learned, hollow = failed level)")
        ax.set_title(title)
    fig.subplots_adjust(wspace=0.08)
    t.to_csv(OUT_DIR / "episodes_to_converge.csv", index=False)
    return _save(fig, "fig9_episodes_to_converge.png")


# ----------------------------------------------------------------- helpers
def _seed_test(algo: str) -> list[float]:
    return [_summary(algo, r)["test"]["mean_return"] for r in _seed_runs(algo)]


_GEN_CACHE: dict | None = None


def _generalization(regen: bool = False) -> dict:
    """Evaluate the selected models + baselines under new conditions (cached)."""
    global _GEN_CACHE
    path = LOGS_DIR / "generalization.json"
    if _GEN_CACHE is not None and not regen:
        return _GEN_CACHE
    if path.exists() and not regen:
        _GEN_CACHE = json.loads(path.read_text(encoding="utf-8"))
        return _GEN_CACHE
    from training.common import (
        TEST_SEEDS,
        VAL_SEEDS,
        argmax_policy,
        evaluate,
        load_policy,
    )

    print("  running generalization evaluation (cached afterwards in logs/generalization.json)...")
    res = _results()
    acts = {a: argmax_policy(load_policy(ROOT / res[a]["model"])[0]) for a in ALGOS}
    strip = lambda r: {k: v for k, v in r.items() if k != "returns"}
    out: dict = {"seeds": "test 1000-1049", "conditions": GEN_CONDITIONS, "results": {}}
    for cond, kw in GEN_CONDITIONS.items():
        row = {a: strip(evaluate(acts[a], TEST_SEEDS, env_kwargs=kw)) for a in ALGOS}
        for b in ("greedy", "random"):
            row[b] = strip(evaluate(None, TEST_SEEDS, baseline=b, env_kwargs=kw))
        out["results"][cond] = row
        print(f"    {cond.replace(chr(10), ' '):<28} " + "  ".join(f"{k} {v['mean_return']:7.1f}" for k, v in row.items()))
    out["val_baselines"] = {b: strip(evaluate(None, VAL_SEEDS, baseline=b)) for b in ("greedy", "random")}
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _GEN_CACHE = out
    return out


def write_tables() -> Path:
    """CSV tables for the report: final comparison + generalization."""
    res, gen = _results(), _generalization()
    rows = []
    for algo in ALGOS:
        t, seeds = res[algo]["test"], _seed_test(algo)
        rows.append({"policy": LABEL[algo], "selected_run": res[algo]["best_run"],
                     "val_return": round(res[algo]["val"]["mean_return"], 1),
                     "test_return": round(t["mean_return"], 1),
                     "test_return_3seed_mean": round(float(np.mean(seeds)), 1),
                     "test_return_3seed_sample_std": round(float(np.std(seeds, ddof=1)), 1),
                     "contained": t["contained_rate"], "village_safe": t["village_safe_rate"],
                     "battery_failed": t["battery_fail_rate"], "land_saved": round(t["mean_land_saved"], 3),
                     "episode_length": t["mean_length"]})
    for b in ("greedy", "random"):
        t = res[f"baseline_{b}"]["test"]
        rows.append({"policy": f"{b} baseline", "selected_run": "-", "val_return": round(gen["val_baselines"][b]["mean_return"], 1),
                     "test_return": round(t["mean_return"], 1), "contained": t["contained_rate"],
                     "village_safe": t["village_safe_rate"], "battery_failed": t["battery_fail_rate"],
                     "land_saved": round(t["mean_land_saved"], 3), "episode_length": t["mean_length"]})
    path = OUT_DIR / "final_results.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    g = [{"condition": c.replace("\n", " "), "policy": p, "mean_return": round(v["mean_return"], 1),
          "contained": v["contained_rate"], "village_safe": v["village_safe_rate"]}
         for c, row in gen["results"].items() for p, v in row.items()]
    pd.DataFrame(g).to_csv(OUT_DIR / "generalization.csv", index=False)
    print(f"  saved {path.relative_to(ROOT)}, assets/plots/generalization.csv")
    return path


def make_all(regen: bool = False) -> None:
    _style()
    _generalization(regen)
    for fn in (fig_comparative_rewards, fig_dqn_objective, fig_pg_entropy, fig_convergence,
               fig_sweep, fig_test_results, fig_generalization, fig_rewards_by_episode,
               fig_episodes_to_converge):
        fn()
    write_tables()


if __name__ == "__main__":
    import sys

    make_all(regen="--regen" in sys.argv)

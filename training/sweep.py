"""Two-phase hyperparameter sweep, robustness re-runs and final model selection.

Phase 1   runs 01-09: baseline + 8 single-setting changes (per algorithm)
Phase 2   runs 10-11: combine the changes of the top-2 / top-3 phase-1 runs
          (ranked on VALIDATION seeds; selection saved to phase2_selection.json)
Robust    best of runs 01-11 retrained with seeds 1 and 2
Finalize  copy best model per algorithm to models/, report TEST-seed metrics
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from training.common import LOGS_DIR, MODELS_DIR, ROOT, TEST_SEEDS, AlgoSpec, evaluate

ALGOS = ("dqn", "reinforce", "ppo", "a2c")
ROBUST_SEEDS = (1, 2)


def get_spec(algo: str) -> AlgoSpec:
    if algo == "dqn":
        from training.dqn_training import SPEC

        return SPEC
    from training.pg_training import SPECS

    return SPECS[algo]


def resolve_algos(algo: str) -> list[str]:
    return list(ALGOS) if algo == "all" else [algo]


# ------------------------------------------------------------------- jobs
GB_PER_WORKER = 0.9  # measured: each torch + torchrl process uses ~0.7 GB


def available_memory_gb() -> float | None:
    try:
        if os.name == "nt":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MemStatus()
            stat.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            # commit headroom limits how many torch processes can start
            return min(stat.ullAvailPhys * 2, stat.ullAvailPageFile) / 1024**3
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except Exception:
        return None


def default_workers() -> int:
    """Parallel runs limited by both CPU cores and free memory."""
    by_cpu = max(1, (os.cpu_count() or 2) - 2)
    mem = available_memory_gb()
    by_mem = max(1, int(mem / GB_PER_WORKER)) if mem is not None else 4
    workers = min(by_cpu, by_mem)
    print(f"Auto workers: {workers} (CPU allows {by_cpu}, free memory ~{mem:.1f} GB allows {by_mem})"
          if mem is not None else f"Auto workers: {workers}")
    return workers


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Executed in a worker process: one training run."""
    import torch

    torch.set_num_threads(1)
    algo, run_name = job["algo"], job["run_name"]
    try:
        spec = get_spec(algo)
        summary = spec.train_fn(job["config"], LOGS_DIR / algo / run_name, run_name)
        return {"algo": algo, "run": run_name, "ok": True, "val": summary["val"]["mean_return"]}
    except Exception:  # keep the sweep alive if one run fails
        err = traceback.format_exc()
        print(f"[{algo}/{run_name}] FAILED\n{err}", flush=True)
        return {"algo": algo, "run": run_name, "ok": False, "error": err}


def run_jobs(jobs: list[dict[str, Any]], workers: int | None, force: bool = False) -> list[dict[str, Any]]:
    todo = []
    for job in jobs:
        done_file = LOGS_DIR / job["algo"] / job["run_name"] / "summary.json"
        if done_file.exists() and not force:
            print(f"skip {job['algo']}/{job['run_name']} (already finished; use --force to rerun)")
            continue
        todo.append(job)
    if not todo:
        return []
    workers = workers or default_workers()
    workers = min(workers, len(todo))
    print(f"Launching {len(todo)} runs on {workers} worker process(es)...", flush=True)
    results = []
    if workers == 1:
        return [_run_job(j) for j in todo]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_run_job, j): j for j in todo}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            status = f"val {res['val']:.1f}" if res["ok"] else "FAILED"
            print(f"==> finished {res['algo']}/{res['run']}: {status} ({len(results)}/{len(todo)})", flush=True)
    return results


def make_job(algo: str, run_name: str, overrides: dict[str, Any], phase: str, seed: int = 0,
             frames: int | None = None) -> dict[str, Any]:
    config = {**overrides, "overrides": overrides, "phase": phase, "seed": seed}
    if frames:
        config["total_frames"] = frames
    return {"algo": algo, "run_name": run_name, "config": config}


# ------------------------------------------------------------------ phases
def phase1_names(spec: AlgoSpec) -> list[tuple[str, dict[str, Any]]]:
    return [(f"run_{i:02d}_{label}", ov) for i, (label, ov) in enumerate(spec.phase1, start=1)]


def load_summary(algo: str, run_name: str) -> dict[str, Any] | None:
    path = LOGS_DIR / algo / run_name / "summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def phase1_jobs(algo: str, frames: int | None) -> list[dict[str, Any]]:
    spec = get_spec(algo)
    return [make_job(algo, name, ov, "phase1", frames=frames) for name, ov in phase1_names(spec)]


def select_phase2(algo: str) -> dict[str, Any]:
    """Rank phase-1 runs on validation return and build the two combinations."""
    spec = get_spec(algo)
    rows = []
    for name, ov in phase1_names(spec):
        s = load_summary(algo, name)
        if s is None:
            raise RuntimeError(f"{algo}/{name} has no summary.json - finish phase 1 first.")
        rows.append({"run": name, "overrides": ov, "val_return": s["val"]["mean_return"],
                     "val_contained": s["val"]["contained_rate"]})
    baseline = rows[0]
    ranked = sorted(rows[1:], key=lambda r: r["val_return"], reverse=True)
    for r in ranked:
        r["beats_baseline"] = r["val_return"] > baseline["val_return"]

    def combine(k: int) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        used, skipped = [], []
        for r in ranked:
            if any(key in merged for key in r["overrides"]):
                skipped.append(r["run"])  # same setting already set by a better run
                continue
            merged.update(r["overrides"])
            used.append(r["run"])
            if len(used) == k:
                break
        return {"overrides": merged, "from_runs": used, "skipped_conflicts": skipped}

    selection = {
        "algo": algo,
        "criterion": "mean return on validation seeds 500-549",
        "baseline": baseline,
        "ranking": ranked,
        "run_10_combo_top2": combine(2),
        "run_11_combo_top3": combine(3),
    }
    out = LOGS_DIR / algo / "phase2_selection.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"\n[{algo}] phase-1 ranking (validation return):  baseline {baseline['val_return']:.1f}")
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r['run']:<28} {r['val_return']:8.1f}  {'(beats baseline)' if r['beats_baseline'] else ''}")
    print(f"  run 10 <- {selection['run_10_combo_top2']['from_runs']}: {selection['run_10_combo_top2']['overrides']}")
    print(f"  run 11 <- {selection['run_11_combo_top3']['from_runs']}: {selection['run_11_combo_top3']['overrides']}")
    return selection


def phase2_jobs(algo: str, frames: int | None) -> list[dict[str, Any]]:
    sel = select_phase2(algo)
    return [
        make_job(algo, "run_10_combo_top2", sel["run_10_combo_top2"]["overrides"], "phase2", frames=frames),
        make_job(algo, "run_11_combo_top3", sel["run_11_combo_top3"]["overrides"], "phase2", frames=frames),
    ]


def sweep_runs(algo: str) -> list[str]:
    spec = get_spec(algo)
    names = [n for n, _ in phase1_names(spec)] + ["run_10_combo_top2", "run_11_combo_top3"]
    return [n for n in names if load_summary(algo, n) is not None]


def best_run(algo: str) -> dict[str, Any]:
    summaries = [load_summary(algo, n) for n in sweep_runs(algo)]
    if not summaries:
        raise RuntimeError(f"No finished sweep runs for {algo}.")
    return max(summaries, key=lambda s: s["val"]["mean_return"])


def robust_jobs(algo: str, frames: int | None) -> list[dict[str, Any]]:
    best = best_run(algo)
    print(f"[{algo}] best sweep run: {best['run']} (val {best['val']['mean_return']:.1f}) -> seeds {ROBUST_SEEDS}")
    with open(LOGS_DIR / algo / "robust_selection.json", "w", encoding="utf-8") as f:
        json.dump({"best_run": best["run"], "overrides": best["overrides"], "seeds": [0, *ROBUST_SEEDS]}, f, indent=2)
    return [make_job(algo, f"robust_seed{s}", best["overrides"], "robust", seed=s, frames=frames) for s in ROBUST_SEEDS]


def run_phase(algo: str, phase: str, workers: int | None, frames: int | None, force: bool) -> None:
    builders = {"1": phase1_jobs, "2": phase2_jobs, "robust": robust_jobs}
    jobs: list[dict[str, Any]] = []
    for a in resolve_algos(algo):
        jobs += builders[phase](a, frames)
    run_jobs(jobs, workers, force)
    for a in resolve_algos(algo):
        write_sweep_table(a)


# ---------------------------------------------------------------- reporting
def write_sweep_table(algo: str) -> None:
    import csv

    names = sweep_runs(algo) + [f"robust_seed{s}" for s in ROBUST_SEEDS]
    rows = []
    for n in names:
        s = load_summary(algo, n)
        if s is None:
            continue
        v, t = s["val"], s["test"]
        rows.append({
            "run": n,
            "changed": json.dumps(s["overrides"]) if s["overrides"] else "baseline",
            "seed": s["config"]["seed"],
            "val_return": round(v["mean_return"], 2),
            "val_std": round(v["std_return"], 2),
            "val_contained": round(v["contained_rate"], 3),
            "val_village_safe": round(v["village_safe_rate"], 3),
            "val_land_saved": round(v["mean_land_saved"], 3),
            "test_return": round(t["mean_return"], 2),
            "test_contained": round(t["contained_rate"], 3),
            "final_train_return": round(s["final_train_return"], 2) if s["final_train_return"] is not None else "",
            "train_min": round(s["train_time_s"] / 60, 1),
        })
    if rows:
        with open(LOGS_DIR / algo / "sweep_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def finalize() -> dict[str, Any]:
    """Pick the best run per algorithm (validation), copy its model, report test metrics."""
    results: dict[str, Any] = {"selection": "best validation return over runs 01-11", "test_seeds": "1000-1049"}
    for algo in ALGOS:
        try:
            best = best_run(algo)
        except RuntimeError as e:
            print(f"[{algo}] {e}")
            continue
        spec = get_spec(algo)
        src = ROOT / best["model_path"]
        dst = MODELS_DIR / spec.group / f"{algo}_best.pt"
        shutil.copyfile(src, dst)
        seeds_test = [best["test"]["mean_return"]]
        for s in ROBUST_SEEDS:
            r = load_summary(algo, f"robust_seed{s}")
            if r is not None:
                seeds_test.append(r["test"]["mean_return"])
        results[algo] = {
            "best_run": best["run"],
            "overrides": best["overrides"],
            "model": str(dst.relative_to(ROOT)),
            "val": {k: v for k, v in best["val"].items() if k != "returns"},
            "test": {k: v for k, v in best["test"].items() if k != "returns"},
            "test_return_across_seeds": {
                "n_seeds": len(seeds_test),
                "mean": float(np.mean(seeds_test)),
                "std": float(np.std(seeds_test)),
            },
        }
        write_sweep_table(algo)
    for name in ("greedy", "random"):
        res = evaluate(None, TEST_SEEDS, baseline=name)
        results[f"baseline_{name}"] = {"test": {k: v for k, v in res.items() if k != "returns"}}
    with open(LOGS_DIR / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print_results_table(results)
    return results


def print_results_table(results: dict[str, Any]) -> None:
    print(f"\n{'policy':<18}{'run':<26}{'test return':>14}{'contained':>11}{'village safe':>14}{'land saved':>12}{'3-seed mean±std':>18}")
    for key, r in results.items():
        if not isinstance(r, dict) or "test" not in r:
            continue
        t = r["test"]
        seeds = r.get("test_return_across_seeds")
        seed_txt = f"{seeds['mean']:.1f}±{seeds['std']:.1f} (n={seeds['n_seeds']})" if seeds else "-"
        print(f"{key:<18}{r.get('best_run', '-'):<26}{t['mean_return']:>8.1f}±{t['std_return']:<5.1f}"
              f"{t['contained_rate']:>10.0%}{t['village_safe_rate']:>14.0%}{t['mean_land_saved']:>12.0%}{seed_txt:>18}")

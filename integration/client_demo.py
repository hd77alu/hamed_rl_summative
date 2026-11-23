"""End-to-end integration demo: the simulator plays the role of a real drone.

Each step, the "drone" serializes its world state to JSON and POSTs it to the inference API;
the API answers with an action, and the drone executes it. Nothing is shared in memory - the
only link between controller and policy is the HTTP/JSON contract.

    uv run main.py client-demo                   # starts a local server on a free port
    uv run main.py client-demo --url http://127.0.0.1:8000   # use a server you started

It also checks that every remote decision matches the local checkpoint's decision.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import numpy as np

from environment.custom_env import FireShieldEnv
from integration.policy import FireShieldPolicy, state_from_env


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def run(url: str | None = None, episodes: int = 3, seed: int = 1000, backend: str = "numpy",
        verbose: bool = True) -> dict:
    server = None
    if url is None:
        from integration.server import make_server

        server = make_server(port=0, backend=backend, quiet=True)  # port 0 = any free port
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        health = _get(f"{url}/health")
        spec = _get(f"{url}/spec")
        print(f"Connected to {url}: {health['algo'].upper()} policy, backend {health['backend']}, "
              f"{spec['obs_dim']} inputs -> {spec['n_actions']} actions")

        from main import best_model_path

        local = FireShieldPolicy.from_checkpoint(best_model_path(health["algo"]))
        env = FireShieldEnv()
        latencies, mismatches, results = [], 0, []
        for ep in range(episodes):
            env.reset(seed=seed + ep)
            done, total, steps = False, 0.0, 0
            while not done:
                state = state_from_env(env)  # what the drone reports
                t0 = time.perf_counter()
                reply = _post(f"{url}/act", {"state": state})
                latencies.append((time.perf_counter() - t0) * 1000)
                if reply["action"] != local.act(env._get_obs())["action"]:
                    mismatches += 1
                _, r, term, trunc, info = env.step(reply["action"])
                total += r
                steps += 1
                done = term or trunc
                if verbose and steps <= 3:
                    conf = f" (confidence {reply['confidence']:.0%})" if "confidence" in reply else ""
                    print(f"  ep {ep + 1} step {steps}: POST /act -> {reply['action_name']}{conf}")
            results.append({"seed": seed + ep, "outcome": info["outcome"], "return": round(total, 1), "steps": steps})
            print(f"  episode {ep + 1} (seed {seed + ep}): {info['outcome'].upper():<16} return {total:7.1f} "
                  f"in {steps} remote decisions")
        lat = np.array(latencies)
        summary = {
            "url": url, "decisions": len(lat), "mismatches_vs_local": mismatches,
            "latency_ms_median": round(float(np.median(lat)), 2), "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
            "episodes": results,
        }
        print(f"{len(lat)} HTTP decisions | round-trip latency median {summary['latency_ms_median']} ms, "
              f"p95 {summary['latency_ms_p95']} ms | remote vs local checkpoint mismatches: {mismatches}")
        return summary
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()

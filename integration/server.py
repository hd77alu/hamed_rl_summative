"""FireShield inference API: a JSON-over-HTTP endpoint a drone controller or frontend can call.

    uv run main.py serve --port 8000

Endpoints
  GET  /health   -> {"status": "ok", "algo": ..., "backend": ...}
  GET  /spec     -> policy card: observation layout, action names, test metrics
  GET  /example  -> a valid world-state request body (seed 1000)
  POST /act      -> body {"state": {...}}  or  {"observation": [65 floats]}
                    reply {"action": 8, "action_name": "fly_to_fire", "confidence": 0.71,
                           "probabilities": {...}, "latency_ms": 0.4}

Uses only the Python standard library (http.server), so it adds no dependencies. Requests are
handled on separate threads; the policy itself is stateless (one decision per request).
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from integration.policy import EXPORT_DIR, FireShieldPolicy, StateError, example_state

MAX_BODY = 1_000_000


def load_default_policy(backend: str = "numpy", algo: str | None = None) -> FireShieldPolicy:
    """numpy/torchscript: models/export/ (run `main.py export` first); checkpoint: models/*/<algo>_best.pt."""
    if backend == "numpy":
        return FireShieldPolicy.from_numpy(EXPORT_DIR / "policy_weights.npz")
    if backend == "torchscript":
        return FireShieldPolicy.from_torchscript(EXPORT_DIR / "policy.torchscript.pt")
    from main import best_model_path

    path = best_model_path(algo)
    if path is None:
        raise FileNotFoundError("No trained model found - train and run `uv run main.py finalize` first.")
    return FireShieldPolicy.from_checkpoint(path)


def make_handler(policy: FireShieldPolicy, backend: str, quiet: bool = False):
    lock = threading.Lock()  # torch modules are not guaranteed thread-safe
    spec = {k: v for k, v in policy.meta.items()}

    class Handler(BaseHTTPRequestHandler):
        server_version = "FireShieldAPI/1.0"

        def _send(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")  # browser frontends
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self) -> None:  # CORS preflight
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok", "algo": policy.meta.get("algo"), "backend": backend})
            elif self.path == "/spec":
                self._send(200, spec)
            elif self.path == "/example":
                self._send(200, {"state": example_state()})
            else:
                self._send(404, {"error": f"unknown endpoint {self.path}"})

        def do_POST(self) -> None:
            if self.path != "/act":
                self._send(404, {"error": f"unknown endpoint {self.path}"})
                return
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_BODY:
                self._send(400, {"error": "request body missing or too large"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                start = time.perf_counter()
                with lock:
                    if "observation" in body:
                        result = policy.act(body["observation"])
                    elif "state" in body:
                        result = policy.act_from_state(body["state"])
                    else:
                        raise StateError("body must contain 'state' or 'observation'")
                result["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
                self._send(200, result)
            except json.JSONDecodeError:
                self._send(400, {"error": "body is not valid JSON"})
            except (StateError, AttributeError, TypeError) as e:
                self._send(422, {"error": str(e)})

        def log_message(self, fmt: str, *args) -> None:
            if not quiet:
                super().log_message(fmt, *args)

    return Handler


def make_server(host: str = "127.0.0.1", port: int = 8000, backend: str = "numpy",
                algo: str | None = None, quiet: bool = False) -> ThreadingHTTPServer:
    policy = load_default_policy(backend, algo)
    return ThreadingHTTPServer((host, port), make_handler(policy, backend, quiet))


def serve(host: str = "127.0.0.1", port: int = 8000, backend: str = "numpy", algo: str | None = None) -> None:
    server = make_server(host, port, backend, algo)
    print(f"FireShield inference API on http://{host}:{port}  (backend: {backend})")
    print("  GET /health   GET /spec   GET /example   POST /act   - Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()

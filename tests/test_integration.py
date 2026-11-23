"""Integration path: state serialization, export formats, and the HTTP inference API.

Uses a freshly initialised network, so these tests do not need trained models.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from environment import FireShieldEnv
from integration.policy import (
    FireShieldPolicy,
    StateError,
    observation_from_state,
    state_from_env,
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    import torch

    from integration.export import export
    from training.common import build_mlp, obs_and_action_dims, save_policy

    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("export")
    obs_dim, n_actions = obs_and_action_dims()
    ckpt = d / "ppo_test.pt"
    save_policy(ckpt, build_mlp(obs_dim, n_actions, 32, 2), "ppo", {"hidden": 32, "depth": 2})
    export(ckpt, d)  # raises if the formats disagree
    return ckpt, d


def test_state_roundtrip_matches_env_observation():
    env = FireShieldEnv()
    rng = np.random.default_rng(0)
    env.reset(seed=1000)
    for _ in range(40):
        state = json.loads(json.dumps(state_from_env(env)))  # through real JSON
        np.testing.assert_allclose(observation_from_state(state), env._get_obs(), atol=1e-6)
        _, _, term, trunc, _ = env.step(int(rng.integers(9)))
        if term or trunc:
            env.reset(seed=1001)


@pytest.mark.parametrize("mutate", [
    lambda s: s.pop("drone"),
    lambda s: s.update(drone=[99, 0]),
    lambda s: s.update(fire_intensity=[[0.0]]),
    lambda s: s.update(water=-1),
    lambda s: s["wind"].update(direction="UP"),
])
def test_bad_state_rejected(mutate):
    env = FireShieldEnv()
    env.reset(seed=3)
    state = state_from_env(env)
    mutate(state)
    with pytest.raises(StateError):
        observation_from_state(state)


def test_export_formats_agree(exported):
    ckpt, d = exported
    pols = [FireShieldPolicy.from_checkpoint(ckpt), FireShieldPolicy.from_torchscript(d / "policy.torchscript.pt"),
            FireShieldPolicy.from_numpy(d / "policy_weights.npz")]
    env = FireShieldEnv()
    obs, _ = env.reset(seed=5)
    outs = [p.act(obs) for p in pols]
    assert len({o["action"] for o in outs}) == 1
    np.testing.assert_allclose(list(outs[0]["probabilities"].values()), list(outs[2]["probabilities"].values()), atol=1e-3)
    card = json.loads((d / "policy_card.json").read_text(encoding="utf-8"))
    assert card["obs_dim"] == 65 and len(card["actions"]) == 9


def test_wrong_observation_size_rejected(exported):
    _, d = exported
    with pytest.raises(StateError):
        FireShieldPolicy.from_numpy(d / "policy_weights.npz").act([0.0] * 10)


def _request(url, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_api(exported):
    from integration.server import make_handler

    _, d = exported
    policy = FireShieldPolicy.from_numpy(d / "policy_weights.npz")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(policy, "numpy", quiet=True))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert _request(f"{url}/health") == (200, {"status": "ok", "algo": "ppo", "backend": "numpy"})
        code, example = _request(f"{url}/example")
        assert code == 200
        code, reply = _request(f"{url}/act", example)
        assert code == 200 and reply["action_name"] == policy.act_from_state(example["state"])["action_name"]
        code, reply = _request(f"{url}/act", {"observation": observation_from_state(example["state"]).tolist()})
        assert code == 200 and 0 <= reply["action"] < 9
        assert _request(f"{url}/act", {"state": {"drone": [0, 0]}})[0] == 422
        assert _request(f"{url}/act", raw=b"not json")[0] == 400
        assert _request(f"{url}/nope")[0] == 404
    finally:
        server.shutdown()
        server.server_close()

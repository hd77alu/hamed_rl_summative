import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment import ENV_ID, FireShieldEnv
from environment.baselines import greedy_policy, random_policy
from environment.custom_env import (
    BASE,
    DROP_WATER,
    FIREBREAK,
    HOVER,
    MOVE_N,
    RETURN_BASE,
    VILLAGE,
)


def run_episode(env, policy, seed):
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(env))
        assert env.observation_space.contains(obs)
        total += r
        done = term or trunc
    return total, info


def test_gymnasium_api_compliance():
    check_env(FireShieldEnv(), skip_render_check=True)


def test_registered_id():
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape


def test_spaces():
    env = FireShieldEnv()
    assert env.action_space.n == 9
    expected = sum(size for _, size, _, _ in env.obs_layout)
    assert env.observation_space.shape == (expected,)
    obs, _ = env.reset(seed=1)
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)


def test_start_state():
    env = FireShieldEnv()
    env.reset(seed=2)
    assert tuple(env.drone) == env.base_pos
    assert env.terrain[env.base_pos] == BASE
    assert env.water == env.cfg.water_capacity
    assert env.battery == env.cfg.battery_capacity
    assert 1 <= (env.intensity > 0).sum() <= env.cfg.max_ignitions
    assert not (env.intensity[env.terrain == VILLAGE] > 0).any()


def test_determinism_under_seed():
    a, b = FireShieldEnv(), FireShieldEnv()
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    np.testing.assert_array_equal(oa, ob)
    rng = np.random.default_rng(0)
    for _ in range(60):
        act = int(rng.integers(9))
        ra = a.step(act)
        rb = b.step(act)
        np.testing.assert_array_equal(ra[0], rb[0])
        assert ra[1:4] == rb[1:4]
        if ra[2] or ra[3]:
            break


def test_drop_water_extinguishes_and_uses_tank():
    env = FireShieldEnv()
    env.reset(seed=3)
    r, c = env.drone
    env.intensity[:] = 0
    env.intensity[r, c] = 0.5
    _, reward, *_ = env.step(DROP_WATER)
    assert env.water == env.cfg.water_capacity - 1
    assert env.cells_extinguished == 1
    assert env.wet[r, c] > 0


def test_wasted_drop_penalized():
    env = FireShieldEnv()
    env.reset(seed=4)
    env.water = 0
    _, reward, *_ = env.step(DROP_WATER)
    assert reward <= env.cfg.r_wasted_drop + env.cfg.r_step + 1e-6 or env.outcome != "running"


def test_firebreak_removes_fuel_and_invalid_on_base():
    env = FireShieldEnv()
    env.reset(seed=5)
    _, reward, *_ = env.step(FIREBREAK)  # drone starts on the base -> invalid
    assert env.firebreaks_built == 0
    env.step(MOVE_N) if env.drone[0] > 0 else None
    r, c = env.drone
    if env.fuel[r, c] > 0 and env.intensity[r, c] == 0:
        env.step(FIREBREAK)
        assert env.firebreak[r, c] and env.fuel[r, c] == 0


def test_refill_at_base():
    env = FireShieldEnv()
    env.reset(seed=6)
    env.water, env.battery = 0, 10
    env.step(RETURN_BASE)
    assert env.water == env.cfg.water_capacity
    assert env.battery == env.cfg.battery_capacity


def test_village_ignition_terminates():
    env = FireShieldEnv(spread_prob=1.0)
    env.reset(seed=7)
    vr, vc = env.village_cells[0]
    env.intensity[:] = 0
    nr = vr + 2 if vr == 0 else vr - 1
    env.fuel[nr, vc] = 1.0  # make sure the neighbour is flammable
    env.intensity[nr, vc] = 1.0
    env.wet[:] = 0
    done, info = False, {}
    for _ in range(10):
        _, _, term, trunc, info = env.step(HOVER)
        if term or trunc:
            break
    assert info["outcome"] == "village_burnt" and info["village_burnt"]


def test_battery_depletion_terminates():
    env = FireShieldEnv(spread_prob=0.0, battery_capacity=3)
    env.reset(seed=8)
    for _ in range(3):
        _, _, term, _, info = env.step(HOVER)
    assert term and info["outcome"] == "battery_depleted"


def test_truncation():
    env = FireShieldEnv(spread_prob=0.0, burn_rate=0.0, max_steps=5, battery_capacity=100)
    env.reset(seed=9)
    for _ in range(5):
        _, _, term, trunc, info = env.step(HOVER)
    assert trunc and not term and info["outcome"] == "timeout"


@pytest.mark.parametrize("seed", range(5))
def test_episodes_finish(seed):
    env = FireShieldEnv()
    _, info = run_episode(env, random_policy, seed)
    assert info["outcome"] in {"contained", "village_burnt", "battery_depleted", "timeout"}


def test_greedy_beats_random():
    env = FireShieldEnv()
    greedy = np.mean([run_episode(env, greedy_policy, s)[0] for s in range(30)])
    rand = np.mean([run_episode(env, random_policy, s)[0] for s in range(30)])
    assert greedy > rand


def test_rgb_render():
    env = FireShieldEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    env.close()


def test_potential_shaping_rewards_approach():
    """Moving toward the nearest fire earns more shaping than moving away."""
    rewards = {}
    for name, step_dir in (("toward", 1), ("away", -1)):
        env = FireShieldEnv(shaping_coef=3.0, spread_prob=0.0)
        env.reset(seed=11)
        env.intensity[:] = 0
        r, c = env.drone
        target_c = c + 3 if c + 3 < env.n else c - 3
        env.fuel[r, target_c] = 1.0
        env.intensity[r, target_c] = 0.5
        toward_east = target_c > c
        action = (2 if toward_east else 3) if step_dir == 1 else (3 if toward_east else 2)
        _, rew, *_ = env.step(action)
        rewards[name] = rew
    assert rewards["toward"] > rewards["away"]


def test_shaping_can_be_disabled():
    env = FireShieldEnv(shaping_coef=0.0)
    env.reset(seed=12)
    env.step(HOVER)
    assert env._get_info()["episode_shaping"] == 0.0


def test_fly_to_fire_moves_closer():
    from environment.custom_env import FLY_TO_FIRE

    env = FireShieldEnv(spread_prob=0.0)
    env.reset(seed=13)
    before = np.abs(env._nearest_fire() - env.drone).sum()
    env.step(FLY_TO_FIRE)
    after = np.abs(env._nearest_fire() - env.drone).sum()
    assert after == before - 1

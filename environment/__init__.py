"""FireShield environment package."""

from gymnasium.envs.registration import register

from environment.custom_env import ACTION_NAMES, FireShieldConfig, FireShieldEnv

ENV_ID = "FireShield-v0"

register(id=ENV_ID, entry_point="environment.custom_env:FireShieldEnv")

__all__ = ["ACTION_NAMES", "ENV_ID", "FireShieldConfig", "FireShieldEnv"]

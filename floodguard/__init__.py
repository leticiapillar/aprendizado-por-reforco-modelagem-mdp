from gymnasium.envs.registration import register

register(
    id="FloodGuard-v0",
    entry_point="floodguard.envs.flood_guard_env:FloodGuardEnv",
)

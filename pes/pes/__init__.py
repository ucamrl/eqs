from gym.envs.registration import register

register(
    id="pes-base-env-v0",
    entry_point="pes.environment.base_env:BaseEnvironment",
)
register(
    id="pes-par-env-v0",
    entry_point="pes.environment.par_env:ParEnvironment",
)

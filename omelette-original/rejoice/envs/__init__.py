from gym.envs.registration import register

register(
    id='egraph-v0',
    entry_point='rejoice.envs.egraph_env:EGraphEnv',
)
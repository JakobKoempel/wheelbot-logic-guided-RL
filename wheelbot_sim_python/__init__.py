from gymnasium.envs.registration import register

register(
    id='wheelbot-v0',
    entry_point='wheelbot_sim_python.envs.wheelbot:WheelbotEnv',
)

register(
    id='wheelbot-v1',
    entry_point='wheelbot_sim_python.envs.wheelbot:WheelbotEnv_V1',
)

register(
    id='wheelbot-v2',
    entry_point='wheelbot_sim_python.envs.wheelbot:WheelbotEnv_V2',
)

register(
    id='wheelbot-v3',
    entry_point='wheelbot_sim_python.envs.wheelbot:WheelbotEnv_V3',
)
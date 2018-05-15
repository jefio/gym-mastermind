from gym.envs.registration import register

register(
    id='Mastermind-v0',
    entry_point='gym_mastermind.envs:MastermindEnv',
)

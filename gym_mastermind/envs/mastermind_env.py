"""Mastermind"""
from collections import Counter
import logging

import gym
from gym.utils import seeding
from gym import spaces


logger = logging.getLogger(__name__)


class MastermindEnv(gym.Env):
    """
    Guess a 4-digits long password where each digit is between 0 and 5.

    After each step the agent is provided with a 4-digits long tuple:
    - '2' indicates that a digit has been correclty guessed at the correct position.
    - '1' indicates that a digit has been correclty guessed but the position is wrong.
    - '0' otherwise.

    The rewards at the end of the episode are:
    0 if the agent's guess is incorrect
    1 if the agent's guess is correct

    The episode terminates after the agent guesses the target or
    12 steps have been taken.
    """
    values = 6
    size = 4
    guess_max = 12

    def __init__(self):
        self.target = None
        self.guess_count = None
        self.observation = None

        self.observation_space = spaces.Tuple(
            [spaces.Discrete(3) for _ in range(self.size)])
        self.action_space = spaces.Tuple(
            [spaces.Discrete(self.values) for _ in range(self.size)])

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_observation(self, action):
        match_idxs = set(idx for idx, ai in enumerate(action) if ai == self.target[idx])
        n_correct = len(match_idxs)
        g_counter = Counter(self.target[idx] for idx in range(self.size) if idx not in match_idxs)
        a_counter = Counter(action[idx] for idx in range(self.size) if idx not in match_idxs)
        n_white = sum(min(g_count, a_counter[k])for k, g_count in g_counter.items())
        return tuple([0] * (self.size - n_correct - n_white) + [1] * n_white + [2] * n_correct)

    def step(self, action):
        assert self.action_space.contains(action)
        self.guess_count += 1
        done = action == self.target or self.guess_count >= self.guess_max
        if done and action == self.target:
            reward = 1
        else:
            reward = 0
        return self.get_observation(action), reward, done, {}

    def reset(self):
        self.target = self.action_space.sample()
        logger.debug("target=%s", self.target)
        self.guess_count = 0
        self.observation = (0,) * self.size
        return self.observation

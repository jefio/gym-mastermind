# gym-mastermind

An OpenAI gym environment for the Mastermind game.

# Installation

```bash
cd gym-mastermind
pip install -e .
```

# Example

```python
import logging
logging.basicConfig(level=logging.DEBUG)

import gym_mastermind
import gym
env = gym.make('Mastermind-v0')

s = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    s, r, done, _ = env.step(action)
    print(action, s, r)
```

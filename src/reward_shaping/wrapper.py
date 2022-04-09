import inspect
from pprint import pprint

import gym


class RewardShapingWrapper(gym.Wrapper):
    # This wrapper is responsible for reward shaping
    # In this process the reward in time step t is calculated as:
    # r`(t) = r(t) - fi(x_t) + gamma * fi(x_(t+1))

    def __init__(self, env: str, gamma: float, fi: callable, fi_t0: float):
        assert isinstance(env, gym.Env), "Environment must be a gym environment"
        assert isinstance(gamma, float) and 0 < gamma < 1, "Gamma parameter must be a float in range (0,1)"
        assert callable(fi), "Fi must be a callable"
        assert isinstance(fi_t0, float), f"Fi(t0) (passed value = {fi_t0}) must be a float not {type(fi_t0)}"

        super().__init__(env)
        self._fi = fi
        self._gamma = gamma
        self._last_fi_value = fi_t0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        fi_value = self._fi(next_state)
        reward = reward - self._last_fi_value + self._gamma * fi_value

        self._last_fi_value = fi_value

        return next_state, reward, done, info

    def reset(self, **kwargs):
        print("Inside RewardShapingWrapper reset")
        pprint(f"Stack: {inspect.stack()}")
        return self.env.reset(**kwargs)

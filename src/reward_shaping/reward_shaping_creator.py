import inspect
import logging
from random import randint

import gym

from src.constants import Constants
from src.reward_shaping.wrapper import RewardShapingWrapper

logger = logging.getLogger(Constants.LOGGER_NAME)


class RewardShapingEnvironmentCreator:
    def __init__(self, env: str, gamma: float, fi: callable, fi_t0: float):
        assert isinstance(env, str), "Environment parameter must be a string"
        assert isinstance(gamma, float) and 0 < gamma < 1, "Gamma parameter must be a float in range (0,1)"
        assert callable(fi), "Fi must be a callable"
        assert isinstance(fi_t0, float), f"Fi(t0) (passed value = {fi_t0}) must be a float not {type(fi_t0)}"

        self._environment_name = env
        self._gamma = gamma
        self._fi = fi
        self._fi_t0 = fi_t0

    def _build_env(self):
        env = gym.make(self._environment_name)
        wrapped_env = RewardShapingWrapper(env, self._gamma, self._fi, self._fi_t0)

        return wrapped_env

    def __call__(self, *args, **kwargs):
        random_id = randint(100, 200)
        stack = inspect.stack()
        print(f"Inside RewardShapingEnvironmentCreator with id {random_id}, num of elements on stack: {len(stack)}")
        for ele in stack:
            print(f"ID({random_id}) - type: {type(ele)} - value: {ele}")

        return self._build_env()

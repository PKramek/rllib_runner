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


class RewardShapingEnvironmentWithDifferentEvaluationEnvironmentCreator(RewardShapingEnvironmentCreator):

    def __init__(self, env: str, gamma: float, fi: callable, fi_t0: float, eval_env: str):
        super().__init__(env, gamma, fi, fi_t0)
        assert isinstance(eval_env, str), "Evaluation environment parameter must be a string"

        if env != eval_env:
            # Sadly this cannot be a warning log, because it is used in another thread
            print("WARNING! - Base environment is different from evaluation environment")

        self._eval_environment_name = eval_env

    def _build_eval_env(self):
        eval_env = gym.make(self._eval_environment_name)

        return eval_env

    def __call__(self, *args, **kwargs):
        # DON'T USE THAT IN PRODUCTION CODE, IT IS A TERRIBLE CODE PRACTICE!!!!
        stack = inspect.stack()

        # This if statement checks where this function is called from, if it is called for creation of evaluation
        # environment, stack size is 16. This value was obtained by debugging.
        if len(stack) == 16:
            print("Building evaluation environment...")
            env = self._build_eval_env()
        else:
            print("Building training environment...")
            env = self._build_env()

        return env

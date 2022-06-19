import inspect
import logging
from random import randint

import gym

from src.constants import Constants
from src.reward_modifier.wrapper import RewardModifierWrapper

logger = logging.getLogger(Constants.LOGGER_NAME)


class RewardModifierEnvironmentCreator:
    def __init__(self, env: str, psi: callable):
        assert isinstance(env, str), "Environment parameter must be a string"
        assert callable(psi), "Fi must be a callable"

        self._environment_name = env
        self._psi = psi

    def _build_env(self):
        env = gym.make(self._environment_name)
        wrapped_env = RewardModifierWrapper(env, self._psi)

        return wrapped_env

    def __call__(self, *args, **kwargs):
        random_id = randint(100, 200)
        stack = inspect.stack()
        print(f"Inside RewardModifierEnvironmentCreator with id {random_id}, num of elements on stack: {len(stack)}")
        for ele in stack:
            print(f"ID({random_id}) - type: {type(ele)} - value: {ele}")

        return self._build_env()


class RewardModifierEnvironmentWithDifferentEvaluationEnvironmentCreator(RewardModifierEnvironmentCreator):

    def __init__(self, env: str, psi: callable, eval_env: str):
        super().__init__(env, psi)
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
        stack_len = len(stack)

        # This if statement checks where this function is called from, if it is called for creation of evaluation
        # environment, stack size is 16. This value was obtained by debugging.
        if stack_len == 16:
            print("Building evaluation environment...")
            env = self._build_eval_env()
        elif stack_len == 6:
            print("Building training environment...")
            env = self._build_env()
        else:
            raise RuntimeError(f"Unsupported case: Stack has length {stack_len}, supported cases include 6 and 16")

        return env

import gym

from src.reward_shaping.wrapper import RewardShapingWrapper


class RewardShapingEnvironmentCreator:
    def __init__(self, environment_name: str, gamma: float, fi: callable):
        self._environment_name = environment_name
        self._gamma = gamma
        self._fi = fi

    def _build_env(self):
        env = gym.make(self._environment_name)
        wrapped_env = RewardShapingWrapper(env, self._gamma, self._fi)

        return wrapped_env

    def __call__(self, *args, **kwargs):
        return self._build_env()

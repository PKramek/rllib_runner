from abc import ABC, abstractmethod
from typing import Dict

import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.sac as sac


def identical_dict_keys(first: Dict, second: Dict) -> bool:
    assert isinstance(first, dict), f'First parameter must be a dictionary not {type(first)}'
    assert isinstance(second, dict), f'Second parameter must be a dictionary not {type(first)}'

    first_keys = first.keys()
    second_keys = second.keys()

    return set(first_keys) == set(second_keys)


class AlgorithmStrategy(ABC):

    def __init__(self):
        self.__config = None
        self.__trainer = None

    @property
    @abstractmethod
    def _rl_lib_algorithm(self):
        pass

    @abstractmethod
    def get_trainer(self, config: Dict, env: str):
        pass

    def get_default_config(self) -> Dict:
        return self._rl_lib_algorithm.DEFAULT_CONFIG.copy()


class PPOStrategy(AlgorithmStrategy):

    @property
    def _rl_lib_algorithm(self):
        return ppo

    def get_trainer(self, config: Dict, env: str):
        return ppo.PPOTrainer(config, env)


class SACStrategy(AlgorithmStrategy):

    @property
    def _rl_lib_algorithm(self):
        return sac

    def get_trainer(self, config: Dict, env: str):
        return sac.SACTrainer(config, env)


class AlgorithmFactory:
    _ALGORITHM_MAPPING = {
        'PPO': PPOStrategy,
        'SAV': SACStrategy
    }

    @staticmethod
    def get_algorithm(algorithm: str) -> AlgorithmStrategy:
        algorithm = AlgorithmFactory._ALGORITHM_MAPPING.get(algorithm, None)

        if algorithm is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return algorithm()

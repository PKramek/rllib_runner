from abc import ABC, abstractmethod
from typing import Dict

from ray.rllib.agents import ppo as ppo, sac as sac

from src.constants import Constants
from src.util import get_sub_dictionary, split_dictionary


class AlgorithmStrategy(ABC):
    @property
    @abstractmethod
    def _rl_lib_algorithm(self):
        pass

    @property
    @abstractmethod
    def algorithm_specific_parameters(self) -> Dict[str, str]:
        pass

    def get_default_config(self) -> Dict[str, str]:
        return self._rl_lib_algorithm.DEFAULT_CONFIG.copy()

    def get_config_from_args_params(self, args_params: Dict) -> Dict:
        algorithm_params_keys = self.algorithm_specific_parameters | Constants.COMMON_PARAMS
        algorithm_params = get_sub_dictionary(args_params, algorithm_params_keys)

        algorithm_params, model_params = split_dictionary(
            algorithm_params,
            algorithm_params_keys & Constants.MODEL_PARAMS)

        # RLlib enforces that model parameters are grouped inside dictionary under 'model' key
        algorithm_params['model'] = dict()

        for model_param, value in model_params.items():
            algorithm_params['model'][model_param] = value

        return algorithm_params



class PPOStrategy(AlgorithmStrategy):
    @property
    def _rl_lib_algorithm(self):
        return ppo

    @property
    def algorithm_specific_parameters(self):
        return Constants.PPO_SPECIFIC_PARAMS


class SACStrategy(AlgorithmStrategy):

    @property
    def _rl_lib_algorithm(self):
        return sac

    @property
    def algorithm_specific_parameters(self):
        return Constants.SAC_SPECIFIC_PARAMS


class AlgorithmFactory:
    _ALGORITHM_MAPPING = {
        'PPO': PPOStrategy,
        'SAC': SACStrategy
    }

    @staticmethod
    def get_algorithm(algorithm: str) -> AlgorithmStrategy:
        algorithm = AlgorithmFactory._ALGORITHM_MAPPING.get(algorithm, None)

        if algorithm is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return algorithm()

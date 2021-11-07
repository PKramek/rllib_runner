from abc import ABC, abstractmethod
from typing import Dict

import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.sac as sac

from src.constants import Constants


class AlgorithmStrategy(ABC):
    @property
    @abstractmethod
    def _rl_lib_algorithm(self):
        pass

    @property
    @abstractmethod
    def algorithm_specific_parameters(self):
        pass

    def get_default_config(self) -> Dict:
        return self._rl_lib_algorithm.DEFAULT_CONFIG.copy()

    def get_config_from_args_params(self, args_params: Dict) -> Dict:
        # TODO refactor this method
        config = dict()

        for common_param in Constants.COMMON_PARAMS:
            if common_param == 'train_batch_size':
                config['sgd_minibatch_size'] = args_params[common_param]
                config['train_batch_size'] = args_params[common_param]
            if common_param == 'fcnet_activation':
                try:
                    config['model'][common_param] = args_params[common_param]
                except KeyError:
                    config['model'] = dict()
                    config['model'][common_param] = args_params[common_param]
            else:
                config[common_param] = args_params[common_param]

        for algorithm_param in self.algorithm_specific_parameters:
            if algorithm_param == 'fcnet_hiddens':
                config['model'][algorithm_param] = args_params[algorithm_param]
            else:
                config[algorithm_param] = args_params[algorithm_param]

        return config


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

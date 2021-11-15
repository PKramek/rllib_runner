from abc import ABC, abstractmethod
from typing import Dict, Set

from ray.rllib.agents import ppo as ppo, sac as sac

from src.constants import Constants
from src.util import get_sub_dictionary, split_dictionary


class AlgorithmStrategy(ABC):
    @property
    @abstractmethod
    def _rl_lib_algorithm(self):
        pass

    @abstractmethod
    def set_model_params(self, algorithm_params: Dict, algorithm_params_keys: Set[str]) -> Dict:
        pass

    @abstractmethod
    def set_optimization_params(self, algorithm_params: Dict, algorithm_params_keys: Set[str]) -> Dict:
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

        algorithm_params = self.set_model_params(algorithm_params, algorithm_params_keys)
        algorithm_params = self.set_optimization_params(algorithm_params, algorithm_params_keys)

        return algorithm_params


class PPOStrategy(AlgorithmStrategy):
    @property
    def _rl_lib_algorithm(self):
        return ppo

    @property
    def algorithm_specific_parameters(self):
        return Constants.PPO_SPECIFIC_PARAMS

    def set_model_params(self, algorithm_params: Dict, algorithm_params_keys: Set[str]) -> Dict:
        model_params_for_algorithm = algorithm_params_keys & Constants.MODEL_PARAMS

        algorithm_params, model_params = split_dictionary(
            algorithm_params, model_params_for_algorithm
        )

        algorithm_params['model'] = dict()
        for model_param, value in model_params.items():
            algorithm_params['model'][model_param] = value

        return algorithm_params

    def set_optimization_params(self, algorithm_params: Dict, algorithm_params_keys: Set[str]) -> Dict:
        return algorithm_params


class SACStrategy(AlgorithmStrategy):

    @property
    def _rl_lib_algorithm(self):
        return sac

    @property
    def algorithm_specific_parameters(self):
        return Constants.SAC_SPECIFIC_PARAMS

    def set_model_params(self, algorithm_params: Dict, algorithm_params_keys: Set[str]) -> Dict:
        model_params_for_algorithm = algorithm_params_keys & Constants.MODEL_PARAMS

        algorithm_params, model_params = split_dictionary(
            algorithm_params, model_params_for_algorithm
        )

        algorithm_params['Q_model'] = dict()
        algorithm_params['policy_model'] = dict()

        algorithm_params['Q_model']['fcnet_hiddens'] = model_params['q_value_layers']
        algorithm_params['Q_model']['fcnet_activation'] = model_params['fcnet_activation']

        algorithm_params['policy_model']['fcnet_hiddens'] = model_params['policy_layers']
        algorithm_params['policy_model']['fcnet_activation'] = model_params['fcnet_activation']

        return algorithm_params

    def set_optimization_params(self, algorithm_params: Dict, algorithm_params_keys: Set[str]) -> Dict:
        optimization_params_for_algorithm = algorithm_params_keys & Constants.OPTIMIZATION_PARAMS

        algorithm_params, optimization_params = split_dictionary(
            algorithm_params,
            optimization_params_for_algorithm
        )

        algorithm_params['optimization'] = dict()
        algorithm_params['optimization']['actor_learning_rate'] = optimization_params['actor_learning_rate']
        algorithm_params['optimization']['critic_learning_rate'] = optimization_params['critic_learning_rate']
        algorithm_params['optimization']['entropy_learning_rate'] = optimization_params['entropy_learning_rate']

        return algorithm_params


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

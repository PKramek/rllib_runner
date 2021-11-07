from pprint import pprint

import ray
from ray import tune

from src.args_parser import parser
from src.util import AlgorithmFactory

if __name__ == "__main__":
    args = vars(parser.parse_args())

    algorithm_name = args.pop('algo')
    max_timesteps = args.pop('max_timesteps')

    algorithm = AlgorithmFactory.get_algorithm(algorithm_name)
    config = algorithm.get_config_from_args_params(args)

    pprint(config)

    ray.init()

    tune.run(
        algorithm_name,
        stop={"timesteps_total": max_timesteps},
        config=config
    )

import os
from pprint import pprint

import ray
from ray import tune

from src.algorithms import AlgorithmFactory
from src.args_parser import parser
from src.util import trial_name_generator, trial_dirname_creator, create_and_save_evaluation_results_file

if __name__ == "__main__":
    args = vars(parser.parse_args())

    algorithm_name = args.pop('algo')
    max_timesteps = args.pop('max_timesteps')

    algorithm = AlgorithmFactory.get_algorithm(algorithm_name)
    config = algorithm.get_config_from_args_params(args)

    pprint(config)

    ray.init()

    result = tune.run(
        algorithm_name,
        stop={"timesteps_total": max_timesteps},
        config=config,
        local_dir=os.environ['TENSORBOARD_LOGS_DIR'],
        trial_dirname_creator=trial_dirname_creator,
        trial_name_creator=trial_name_generator
    )

    evaluation_results_df = create_and_save_evaluation_results_file()
    print(evaluation_results_df)

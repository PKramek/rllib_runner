import logging

import ray
from ray import tune

from src.algorithms import AlgorithmFactory
from src.args_parser import parser
from src.constants import Constants
from src.util import trial_name_generator, trial_dirname_creator, create_and_save_evaluation_results_file, setup_logger, \
    add_tune_specific_config_fields

if __name__ == "__main__":
    logger = setup_logger(Constants.LOGGER_NAME)
    args = vars(parser.parse_args())

    algorithm_name = args.pop('algo')
    max_timesteps = args.pop('max_timesteps')

    algorithm = AlgorithmFactory.get_algorithm(algorithm_name)
    config = algorithm.get_config_from_args_params(args)

    config = add_tune_specific_config_fields(config)

    logging.info(f"Using configuration: {config}")

    ray.init()

    result = tune.run(
        algorithm_name,
        stop={"timesteps_total": max_timesteps},
        config=config,
        local_dir=Constants.LOGS_DIRECTORY,
        trial_dirname_creator=trial_dirname_creator,
        trial_name_creator=trial_name_generator
    )

    evaluation_results_df = create_and_save_evaluation_results_file()
    logging.info(f"Evaluation results df:\n{evaluation_results_df}")

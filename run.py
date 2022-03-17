import logging

import ray
from ray import tune
from ray.tune.registry import register_env

from src.algorithms import AlgorithmFactory
from src.args_parser import parser
from src.constants import Constants
from src.reward_shaping.fi import FiFactory
from src.reward_shaping.reward_shaping_creator import RewardShapingEnvironmentCreator
from src.util import trial_name_generator, trial_dirname_creator, create_and_save_evaluation_results_file, setup_logger, \
    add_tune_specific_config_fields, get_max_memory_size

if __name__ == "__main__":

    logger = setup_logger(Constants.LOGGER_NAME)
    args = vars(parser.parse_args())

    fi_x = FiFactory.get_fi("default")
    environment_with_reward_shaping = RewardShapingEnvironmentCreator(
        "CartPole-v1", args["gamma"],
        fi_x,
        fi_x([1.4]))
    register_env("RewardShapingHumanoid-v2", environment_with_reward_shaping)

    algorithm_name = args.pop('algo')
    max_timesteps = args.pop('max_timesteps')

    algorithm = AlgorithmFactory.get_algorithm(algorithm_name)

    config = algorithm.get_config_from_args_params(args)
    config = add_tune_specific_config_fields(config)
    logger.info(f"Using configuration: {config}")

    max_memory_size = get_max_memory_size()

    if max_memory_size is None:
        ray.init()
        logger.warning("Max memory size not set!")
    else:
        ray.init(object_store_memory=max_memory_size)
        logger.warning(f"Max memory size set to: {max_memory_size}")

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

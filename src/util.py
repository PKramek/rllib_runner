import logging
import os
from datetime import datetime
from typing import Dict, Iterable, Tuple

import pandas as pd
from ray import tune

from src.constants import Constants

rllib_runner_logger = logging.getLogger(Constants.LOGGER_NAME)


def split_dictionary(dictionary: Dict, keys: Iterable) -> Tuple[Dict, Dict]:
    """
    Splits dictionary into two. Key, value pairs, where key is in keys iterable will be in one resulting dictionary and
    all the rest of the pairs will be in second dictionary

    :param dictionary: Dictionary to be split
    :type dictionary: Dict
    :param keys: Keys used to split dictionary. All the key, value pairs with key in that iterable will be in only one
    resulting dictionary
    :type keys: Iterable
    :return: Two dictionaries, first contains all the key, value pairs where key was not preset in given keys iterable
    :rtype: Tuple[Dict, Dict]
    """
    assert set(keys) <= set(dictionary.keys()), 'All keys must be in original dictionary'

    without_keys = {key: value for (key, value) in dictionary.items() if key not in keys}
    with_keys = {key: value for (key, value) in dictionary.items() if key in keys}

    return without_keys, with_keys


def get_sub_dictionary(dictionary: Dict, keys: Iterable) -> Dict:
    """
    Return dictionary containing only those key, value pairs which have key in keys iterable

    :param dictionary: Dictionary to get sub dictionary from
    :type dictionary: Dict
    :param keys: Subset of original dictionary keys
    :type keys: Iterable
    :return: Sub dictionary
    :rtype: Dictionary
    """
    assert set(keys) <= set(dictionary.keys()), "All keys must be in original dictionary"

    with_keys = {key: value for (key, value) in dictionary.items() if key in keys}

    return with_keys


def trial_name_generator(trial: tune.trial.Trial) -> str:
    """
    Generates name of the single trial for RLLib`s tune.run function

    :param trial: trial object
    :return: trial name for tune.run function
    """
    logs_dir = Constants.LOGS_DIRECTORY
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    environment_name = trial.config['env']

    trial_name = f"{environment_name}_{trial.trainable_name}_{dt_string}"
    progress_path = f"{logs_dir}/{trial.trainable_name}/{trial_name}/progress.csv"
    eval_results_dir = f"{logs_dir}/{trial.trainable_name}/{trial_name}"

    rllib_runner_logger.warning(
        f"Setting {Constants.ENV_EVALUATION_RESULTS_DIR_PATH} env variable to: {eval_results_dir}")
    os.environ[Constants.ENV_EVALUATION_RESULTS_DIR_PATH] = eval_results_dir

    rllib_runner_logger.warning(f"Setting {Constants.ENV_PROGRESS_FILE_PATH} env variable to: {progress_path}")
    os.environ[Constants.ENV_PROGRESS_FILE_PATH] = progress_path

    rllib_runner_logger.warning(f"Results will be saved in: {logs_dir}/{trial.trainable_name}/{trial_name}")

    return trial_name


def trial_dirname_creator(trial: tune.trial.Trial):
    """
    Generates name of the directory for RLLib`s tune.run function

    :param trial: trial object
    :return: directory name for tune.run function
    """
    return str(trial)


def get_eval_results_df_from_progress_df(progress_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates evaluation results dataframe containing information about evaluation runs results

    :param progress_df: Dataframe created from progress.csv file
    :return: Dataframe containing metrics about evaluation runs from training process
    """
    columns_mapping = {
        'timesteps_total': 'time_step',
        'episode_reward_max': 'eval_return_max',
        'episode_reward_min': 'eval_return_min',
        'episode_reward_mean': 'eval_return_mean',
        'episode_len_mean': 'eval_episode_len_mean'
    }
    evaluation_results_df = progress_df[columns_mapping.keys()]
    evaluation_results_df = evaluation_results_df.rename(columns=columns_mapping)

    return evaluation_results_df


def create_and_save_evaluation_results_file() -> pd.DataFrame:
    """
    Creates evaluation results dataframe and saves it in a file named evaluation_results.csv inside training results
    directory

    :return: Dataframe containing metrics about evaluation runs from training process
    """
    progress_path = os.getenv(Constants.ENV_PROGRESS_FILE_PATH)
    if progress_path is None:
        raise RuntimeError(f"{Constants.ENV_PROGRESS_FILE_PATH} environment variable not set!")

    eval_results_path = os.getenv(Constants.ENV_EVALUATION_RESULTS_DIR_PATH)
    if eval_results_path is None:
        raise RuntimeError(f"{Constants.ENV_EVALUATION_RESULTS_DIR_PATH} environment variable not set!")

    progress_df = pd.read_csv(progress_path)

    evaluation_results_df = get_eval_results_df_from_progress_df(progress_df)
    evaluation_results_df.to_csv(path_or_buf=f"{eval_results_path}/results.csv")

    # Do not remove this logging call, it is used by log parser inside Airflow
    rllib_runner_logger.warning(f"saved evaluation results in {eval_results_path}")

    return evaluation_results_df


def add_tune_specific_config_fields(config: Dict) -> Dict:
    config['num_workers'] = 2

    return config


def setup_logger(logger_name, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(streamHandler)

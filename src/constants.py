class Constants:
    ALGORITHMS = {'PPO', 'SAC'}
    COMMON_PARAMS = {
        'env',
        'gamma',
        'fcnet_activation',
        'fcnet_hiddens',
        'train_batch_size',
        'evaluation_interval'
    }

    MODEL_PARAMS = {
        'fcnet_activation',
        'fcnet_hiddens',
        'policy_layers',
        'q_value_layers',
        'policy_layers',
    }

    PPO_SPECIFIC_PARAMS = {
        'clip_param',
        'lambda',
        'num_sgd_iter',
        'sgd_minibatch_size',
        'vf_clip_param',
        'kl_target',
        'clip_param',
        'lr'
    }

    SAC_SPECIFIC_PARAMS = {
        'actor_learning_rate',
        'critic_learning_rate',
        'entropy_learning_rate',
        'policy_layers',
        'q_value_layers',
        'initial_alpha',
        'buffer_size',
        'tau',
        'learning_starts',
    }

    OPTIMIZATION_PARAMS = {
        'actor_learning_rate',
        'critic_learning_rate',
        'entropy_learning_rate'
    }

    # Env variables used in evaluation results data extraction
    ENV_PROGRESS_FILE_PATH = 'PROGRESS_FILE_PATH'
    ENV_EVALUATION_RESULTS_DIR_PATH = 'EVALUATION_RESULTS_FILE_PATH'
    LOGS_DIRECTORY = '/home/peter/Programowanie/Magisterka/TEMP'
    LOGGER_NAME = 'rllib_runner_logger'

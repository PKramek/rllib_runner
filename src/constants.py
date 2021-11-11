class Constants:
    ALGORITHMS = {'PPO', 'SAC'}
    COMMON_PARAMS = {
        'env',
        'gamma',
        'fcnet_activation',
        'train_batch_size',
        'evaluation_interval'
    }

    MODEL_PARAMS = {
        'fcnet_activation',
        'policy_layers',
        'q_value_layers',
        'fcnet_hiddens'
    }

    PPO_SPECIFIC_PARAMS = {
        'fcnet_hiddens',
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
        'actor_lr',
        'critic_lr',
        'policy_layers',
        'q_value_layers',
        'initial_alpha',
        'buffer_size',
        'tau',
        'learning_starts',
    }

    # Env variables used in evaluation results data extraction
    ENV_PROGRESS_FILE_PATH = 'PROGRESS_FILE_PATH'
    ENV_EVALUATION_RESULTS_DIR_PATH = 'EVALUATION_RESULTS_FILE_PATH'
    LOGS_DIRECTORY = '/tensorboard_logs'
    LOGGER_NAME = 'rllib_runner_logger'

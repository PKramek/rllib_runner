import argparse

from src.constants import Constants

parser = argparse.ArgumentParser(description='CLI based runner for RLLib')
parser.add_argument('--algo', type=str, help='Algorithm to be used', required=True, choices=Constants.ALGORITHMS)
parser.add_argument('--env', type=str, help='OpenAI Gym environment name', default="Humanoid-v2")
parser.add_argument('--max_timesteps', type=int, help='Maximum number of timesteps', default=int(1e6))
parser.add_argument('--gamma', type=float, help='Discount factor', required=False, default=0.99)
parser.add_argument('--initial_alpha', type=float, help='This is the inverse of reward scale', required=False,
                    default=1)
parser.add_argument('--lambda', type=float, help='The GAE (lambda) parameter', required=False, default=0.95)
parser.add_argument('--lr', type=float, help='Learning rate (used in PPO)', required=False, default=0.001)
parser.add_argument('--actor_learning_rate', type=float, help='BaseActor learning rate', required=False, default=0.0003)
parser.add_argument('--critic_learning_rate', type=float, help='Critic learning rate', required=False, default=0.0003)
parser.add_argument('--entropy_learning_rate', type=float, help='Critic learning rate', required=False, default=0.0003)
parser.add_argument('--learning_starts', type=int, help='Experience replay warm start coefficient', default=10000)
parser.add_argument('--tau', type=float, help='Target smoothing coefficient', required=False, default=0.005)
parser.add_argument('--buffer_size', type=int, help='Memory buffer size', required=False, default=int(1e6))
parser.add_argument('--fcnet_hiddens', nargs='+', type=int, help='List of neural network hidden layers sizes',
                    required=False, default=[256, 256])
parser.add_argument('--policy_layers', nargs='+', type=int,
                    help='List of Policy\'s neural network hidden layers sizes',
                    required=False, default=[256, 256])
parser.add_argument('--q_value_layers', nargs='+', type=int,
                    help='List of Q-Value neural network hidden layers sizes',
                    required=False, default=[256, 256])
parser.add_argument('--num_sgd_iter', type=int, help='Number of SGD iterations in PPO learning', default=10)
parser.add_argument('--train_batch_size', type=int, help='Minibatch size, corresponds to horizon size in PPO algorithm',
                    default=64)
parser.add_argument('--sgd_minibatch_size', type=int, help='Minibatch size in PPO',
                    default=64)

# TODO add Target update interval
# parser.add_argument('--n_step', type=int, help='Target update interval', default=64)
parser.add_argument('--evaluation_interval', help='Number of training iterations between evaluations. None disables'
                                                  'evaluation', type=int, default=5)
parser.add_argument('--evaluation_num_episodes', help='Number of evaluation runs', type=int, default=10)
parser.add_argument('--fcnet_activation', type=str, help='Activation function used in hidden layers',
                    default="tanh", choices={'tanh', 'relu'})
parser.add_argument('--kl_target', type=float, help='Target KL in PPO',
                    default=0.01)
parser.add_argument('--vf_clip_param', type=float, help='Value function clipping parameter',
                    default=10.0)
parser.add_argument('--clip_param', type=float, help='Clip parameter used in PPO',
                    default=0.2)

# TODO Add tensorboard support and parametrization
# TODO Add CPU/GPU choosing method

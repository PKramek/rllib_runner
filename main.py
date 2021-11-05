import ray
from ray.tune.logger import pretty_print

from src.util import AlgorithmFactory

ray.init()
alg = 'PPO'

algorithm = AlgorithmFactory.get_algorithm(alg)
config = algorithm.get_default_config()
config["num_gpus"] = 0
config["num_workers"] = 1
trainer = algorithm.get_trainer(config=config, env="CartPole-v0")

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

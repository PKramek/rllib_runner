import gym


class RewardShapingWrapper(gym.Wrapper):
    # This wrapper is responsible for reward shaping
    # In this process the reward in time step t is calculated as:
    # r`(t) = r(t) - fi(x_t) + gamma * fi(x_t+1)

    def __init__(self, env: str, gamma: float, fi: callable):
        super().__init__(env)
        self._fi = fi
        self._gamma = gamma
        self._last_fi_value = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        fi_value = self._fi(next_state)

        reward = reward - self._last_fi_value + self._gamma * fi_value

        self._last_fi_value = fi_value

        return next_state, reward, done, info
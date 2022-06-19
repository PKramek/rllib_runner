import gym


class RewardModifierWrapper(gym.Wrapper):
    # This wrapper is responsible for reward modifying
    # In this process the reward in time step t is calculated as:
    # r`(t) = r(t) + psi(x)

    def __init__(self, env: str, psi: callable):
        assert hasattr(env, 'step') and hasattr(env, 'reset'), "Environment must have step and reset methods"
        assert callable(psi), "Psi must be a callable"

        super().__init__(env)
        self._psi = psi

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        psi_value = self._psi(next_state)
        reward = reward + psi_value

        return next_state, reward, done, info

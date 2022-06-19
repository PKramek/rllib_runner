import pytest
from gym.envs.classic_control import CartPoleEnv
from pytest_mock import MockerFixture

from src.reward_modifier.wrapper import RewardModifierWrapper


class TestRewardModifierWrapper:

    # rt_prim = rt - fi_xt + gamma * fi_xt_plus_one

    @pytest.mark.parametrize("unmodified_reward, psi_value, expected_reward", [
        (0.0, 1.0, 1.0),
        (-1.0, 1.0, 0.0),
        (1, 0, 1),
    ])
    def test_correctly_shapes_reward(self, unmodified_reward: float, psi_value: float,
                                     expected_reward: float, mocker: MockerFixture):
        # Mocks are necessary, because environments are non-deterministic
        psi = mocker.MagicMock(return_value=psi_value)
        env_mock = mocker.MagicMock(spec=CartPoleEnv)
        env_mock.step.return_value = ([0.0], unmodified_reward, False, "")

        wrapped_env = RewardModifierWrapper(env=env_mock, psi=psi)

        _, modified_reward, _, _ = wrapped_env.step(action=[1.0])

        assert modified_reward == pytest.approx(expected_reward)

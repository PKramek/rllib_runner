import pytest
from gym.envs.classic_control import CartPoleEnv
from pytest_mock import MockerFixture

from src.reward_shaping.wrapper import RewardShapingWrapper


class TestRewardShapingWrapper:

    # rt_prim = rt - fi_xt + gamma * fi_xt_plus_one

    @pytest.mark.parametrize("unmodified_reward, last_fi_value, fi_value, gamma, expected_reward", [
        (0.0, 0.0, 0.0, 0.99, 0.0),  # 0.0 - 0.0 + 0.99 * 0.0 = 0.0
        (0.0, 0.0, 1.0, 0.99, 0.99),  # 0 - 0.0 + 0.99 * 1.0 = 0.99
        (0.0, 1.0, 0.0, 0.99, -1.0),  # 0 - 1.0 + 0.99 * 0 = -1.0
        (0.0, 0.0, -100.0, 0.99, -99.0),  # 0 - 0.0 + 0.99 * (-100.0) = -99.0
        (0.0, -100.0, 0.0, 0.99, 100.0),  # 0 - (-100.0) + 0.99 * 0.0 = 100.0
        (0.0, -100.0, -200.0, 0.99, -98.0),  # 0 - (-100.0) + 0.99 * (-200.0) = -98.0
        (0.0, 1000.0, 1000.0, 0.99, -10.0),  # 0 - 1000.0 + 0.99 * 1000.0 = -10
        (0.0, 0.01, 0.01, 0.99, -0.0001),  # 0.0 - 0.01 + 0.99 * 0.01 = -0.0001
        (100.0, 10.0, 100.0, 0.99, 189.0),  # 100.0 - 10.0 + 0.99 * 100.0 = 189.0
        (100.0, 100.0, 10.0, 0.99, 9.9),  # 100.0 - 100.0 + 0.99 * 10.0 = 9.9
        (100.0, 0.0, 10.0, 0.99, 109.9),  # 100.0 - 0.0 + 0.99 * 10.0 = 109.9
        (100.0, 0.0, -10.0, 0.99, 90.1),  # 100.0 - 0.0 + 0.99 * (-10.0) = 90.1
        (100.0, 10.0, 0.0, 0.99, 90.0),  # 100.0 - 10.0 + 0.99 * 0.0 = 90.0
        (100.0, -10.0, 0.0, 0.99, 110.0),  # 100.0 - (-10.0) + 0.99 * 0.0 = 110.0
        (100.0, 0.0, -10.0, 0.9, 91.0),  # 100.0 - 0.0 + 0.9 * (-10.0) = 91.0
    ])
    def test_correctly_shapes_reward(self, unmodified_reward: float, last_fi_value: float,
                                     fi_value: float, gamma: float,
                                     expected_reward: float, mocker: MockerFixture):
        # Mocks are necessary, because environments are non-deterministic
        fi_mock = mocker.MagicMock(return_value=fi_value)
        env_mock = mocker.MagicMock(spec=CartPoleEnv)
        env_mock.step.return_value = ([0.0], unmodified_reward, False, "")

        wrapped_env = RewardShapingWrapper(env=env_mock, gamma=gamma, fi=fi_mock, fi_t0=last_fi_value)

        _, modified_reward, _, _ = wrapped_env.step(action=[1.0])

        assert modified_reward == pytest.approx(expected_reward)

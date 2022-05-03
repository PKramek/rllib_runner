import numpy as np

from src.constants import Constants
from src.reward_shaping.fi_base import Fi


class HumanoidFi(Fi):
    @staticmethod
    def _normal_dist_density(x: float, mean: float, sd: float):
        prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
        return prob_density

    def __call__(self, state):
        # Value of density function is multiplied by 2000, so that its highest possible value is around 300
        index = Constants.HEIGHT_INDEX
        return 2000 * HumanoidFi._normal_dist_density(state[index], 1.4, 0.05)


class HumanoidQuadraticFi(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return -np.power((1.4 - state[index]) * 100, 2)


class HumanoidFlatTopQuadraticFi(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.35 < state[index] < 1.45 else -np.power((1.4 - state[index]) * 100, 2)


class HumanoidNarrowFlatTopQuadraticFi(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.375 < state[index] < 1.425 else -np.power((1.4 - state[index]) * 100, 2)


class HumanoidVeryNarrowFlatTopQuadraticFi(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.39 < state[index] < 1.41 else -np.power((1.4 - state[index]) * 100, 2)


class HumanoidWideFlatTopQuadraticFi(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.3 < state[index] < 1.5 else -np.power((1.4 - state[index]) * 100, 2)


class HumanoidWideFlatTopQuadraticBigPenaltyFi(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.3 < state[index] < 1.5 else -np.power((1.4 - state[index]) * 1000, 2)


class HumanoidWideFlatTopQuadraticSmallPenaltyFi(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.3 < state[index] < 1.5 else -np.power((1.4 - state[index]) * 50, 2)


class HumanoidWideFlatTopQuadraticWithBodyTiltFi(Fi):
    def _height_penalty(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.3 < state[index] < 1.5 else -np.power((1.4 - state[index]) * 100, 2)

    def _forward_tilt_penalty(self, state):
        index = Constants.TILT_INDEX
        return 0.0 if -0.15 < state[index] < 0.15 else -np.power((state[index]) * 100, 2)

    def _x_axis_angle_rotation_penalty(self, state):
        index = Constants.X_AXIS_ROTATION_INDEX
        return 0.0 if -0.05 < state[index] < 0.05 else -np.power((state[index]) * 100, 2)

    def __call__(self, state):
        return self._height_penalty(state) + self._forward_tilt_penalty(state) + self._x_axis_angle_rotation_penalty(
            state)


class HumanoidWideFlatTopQuadraticWithBodyTiltLowerPenaltyFi(Fi):
    def _height_penalty(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.3 < state[index] < 1.5 else -np.power((1.4 - state[index]) * 100, 2)

    def _forward_tilt_penalty(self, state):
        index = Constants.TILT_INDEX
        return 0.0 if -0.15 < state[index] < 0.15 else -np.power((state[index]) * 50, 2)

    def _x_axis_angle_rotation_penalty(self, state):
        index = Constants.X_AXIS_ROTATION_INDEX
        return 0.0 if -0.1 < state[index] < 0.1 else -np.power((state[index]) * 50, 2)

    def __call__(self, state):
        return self._height_penalty(state) + self._forward_tilt_penalty(state) + self._x_axis_angle_rotation_penalty(
            state)


class HumanoidWideFlatTopQuadraticWithForwardBodyTiltLowerPenaltyFi(
    HumanoidWideFlatTopQuadraticWithBodyTiltLowerPenaltyFi):
    def _forward_tilt_penalty(self, state):
        index = Constants.TILT_INDEX
        return 0.0 if 0.05 < state[index] < 0.15 else -np.power((state[index]) * 50, 2)


class HumanoidWideFlatTopQuadraticWithForwardBodyTiltAndCorrectLowerPenaltyFi(
    HumanoidWideFlatTopQuadraticWithBodyTiltLowerPenaltyFi):
    def _forward_tilt_penalty(self, state):
        index = Constants.TILT_INDEX
        return 0.0 if 0.05 < state[index] < 0.15 else -np.power((0.1 - state[index]) * 50, 2)


class HumanoidWideFlatTopQuadraticWithCorrectBodyTiltPenaltyFi(
    HumanoidWideFlatTopQuadraticWithBodyTiltLowerPenaltyFi):
    def _height_penalty(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.3 < state[index] < 1.5 else -np.power((1.4 - state[index]) * 100, 2)

    def _forward_tilt_penalty(self, state):
        index = Constants.TILT_INDEX
        return 0.0 if -0.15 < state[index] < 0.15 else -np.power((state[index]) * 100, 2)

    def _x_axis_angle_rotation_penalty(self, state):
        index = Constants.X_AXIS_ROTATION_INDEX
        return 0.0 if -0.1 < state[index] < 0.1 else -np.power((state[index]) * 100, 2)


class HumanoidWideFlatTopQuadraticWithBodyTiltWide(Fi):
    def _height_penalty(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.3 < state[index] < 1.5 else -np.power((1.4 - state[index]) * 100, 2)

    def _forward_tilt_penalty(self, state):
        index = Constants.TILT_INDEX
        return 0.0 if -0.2 < state[index] < 0.2 else -np.power((state[index]) * 50, 2)

    def _x_axis_angle_rotation_penalty(self, state):
        index = Constants.X_AXIS_ROTATION_INDEX
        return 0.0 if -0.2 < state[index] < 0.2 else -np.power((state[index]) * 50, 2)

    def __call__(self, state):
        return self._height_penalty(state) + self._forward_tilt_penalty(state) + self._x_axis_angle_rotation_penalty(
            state)


class HumanoidVeryWideFlatTopQuadraticFi(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.2 < state[index] < 1.6 else -np.power((1.4 - state[index]) * 100, 2)


class QuadraticHeight(Fi):
    def _height_penalty(self, state):
        index = Constants.HEIGHT_INDEX
        penalty = -np.power((1.4 - state[index]) * 100, 2)
        return penalty

    def __call__(self, state):
        return self._height_penalty(state)


class QuadraticHeightWithForwardTilt(Fi):
    def _height_penalty(self, state):
        index = Constants.HEIGHT_INDEX
        penalty = -np.power((1.4 - state[index]) * 100, 2)
        return penalty

    def _forward_tilt_penalty(self, state):
        index = Constants.TILT_INDEX
        return -np.power((state[index]) * 100, 2)

    def _x_axis_angle_rotation_penalty(self, state):
        index = Constants.X_AXIS_ROTATION_INDEX
        return -np.power((state[index]) * 100, 2)

    def __call__(self, state):
        return self._height_penalty(state) + self._forward_tilt_penalty(state) + self._x_axis_angle_rotation_penalty(
            state)

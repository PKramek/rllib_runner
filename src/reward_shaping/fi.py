from abc import ABC, abstractmethod
from typing import Type

import numpy as np


class Fi(ABC):
    @abstractmethod
    def __call__(self, state):
        pass


# This class is just an example and is not useful in any way
class SumFi(Fi):
    def __call__(self, state):
        return np.sum(state)


class HumanoidFi(Fi):
    @staticmethod
    def _normal_dist_density(x: float, mean: float, sd: float):
        prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
        return prob_density

    def __call__(self, state):
        # Value of density function is multiplied by 2000, so that its highest possible value is around 300
        return 2000 * HumanoidFi._normal_dist_density(state[0], 1.4, 0.05)


class HumanoidQuadraticFi(Fi):
    def __call__(self, state):
        return -np.power((1.4 - state[0]) * 100, 2)


class HumanoidFlatTopQuadraticFi(Fi):
    def __call__(self, state):
        return 0.0 if 1.35 < state[0] < 1.45 else -np.power((1.4 - state[0]) * 100, 2)


class HumanoidNarrowFlatTopQuadraticFi(Fi):
    def __call__(self, state):
        return 0.0 if 1.375 < state[0] < 1.425 else -np.power((1.4 - state[0]) * 100, 2)


class HumanoidVeryNarrowFlatTopQuadraticFi(Fi):
    def __call__(self, state):
        return 0.0 if 1.39 < state[0] < 1.41 else -np.power((1.4 - state[0]) * 100, 2)


class HumanoidWideFlatTopQuadraticFi(Fi):
    def __call__(self, state):
        return 0.0 if 1.3 < state[0] < 1.5 else -np.power((1.4 - state[0]) * 100, 2)


class HumanoidWideFlatTopQuadraticBigPenaltyFi(Fi):
    def __call__(self, state):
        return 0.0 if 1.3 < state[0] < 1.5 else -np.power((1.4 - state[0]) * 1000, 2)


class HumanoidWideFlatTopQuadraticSmallPenaltyFi(Fi):
    def __call__(self, state):
        return 0.0 if 1.3 < state[0] < 1.5 else -np.power((1.4 - state[0]) * 50, 2)


class HumanoidWideFlatTopQuadraticWithBodyTiltFi(Fi):
    def _height_penalty(self, state):
        index = 0
        return 0.0 if 1.3 < state[index] < 1.5 else -np.power((1.4 - state[index]) * 100, 2)

    def _forward_tilt_penalty(self, state):
        index = 3  # In qpos its under index 5, but observation cuts first two elements
        return 0.0 if -0.15 < state[index] < 0.15 else -np.power((state[index]) * 100, 2)

    def _x_axis_angle_rotation_penalty(self, state):
        index = 3  # In qpos its under index 6, but observation cuts first two elements
        return 0.0 if -0.05 < state[index] < 0.05 else -np.power((state[index]) * 100, 2)

    def __call__(self, state):
        return self._height_penalty(state) + self._forward_tilt_penalty(state) + self._x_axis_angle_rotation_penalty(
            state)


# Current best
class HumanoidWideFlatTopQuadraticWithBodyTiltLowerPenaltyFi(Fi):
    def _height_penalty(self, state):
        index = 0
        return 0.0 if 1.3 < state[index] < 1.5 else -np.power((1.4 - state[index]) * 100, 2)

    def _forward_tilt_penalty(self, state):
        index = 3  # In qpos its under index 5, but observation cuts first two elements
        return 0.0 if -0.15 < state[index] < 0.15 else -np.power((state[index]) * 50, 2)

    def _x_axis_angle_rotation_penalty(self, state):
        index = 3  # In qpos its under index 6, but observation cuts first two elements
        return 0.0 if -0.1 < state[index] < 0.1 else -np.power((state[index]) * 50, 2)

    def __call__(self, state):
        return self._height_penalty(state) + self._forward_tilt_penalty(state) + self._x_axis_angle_rotation_penalty(
            state)


class HumanoidWideFlatTopQuadraticWithForwardBodyTiltLowerPenaltyFi(
    HumanoidWideFlatTopQuadraticWithBodyTiltLowerPenaltyFi):
    def _forward_tilt_penalty(self, state):
        index = 3  # In qpos its under index 5, but observation cuts first two elements
        return 0.0 if 0.05 < state[index] < 0.15 else -np.power((state[index]) * 50, 2)


class HumanoidWideFlatTopQuadraticWithBodyTiltWide(Fi):
    def _height_penalty(self, state):
        index = 0
        return 0.0 if 1.3 < state[index] < 1.5 else -np.power((1.4 - state[index]) * 100, 2)

    def _forward_tilt_penalty(self, state):
        index = 3  # In qpos its under index 5, but observation cuts first two elements
        return 0.0 if -0.2 < state[index] < 0.2 else -np.power((state[index]) * 50, 2)

    def _x_axis_angle_rotation_penalty(self, state):
        index = 3  # In qpos its under index 6, but observation cuts first two elements
        return 0.0 if -0.2 < state[index] < 0.2 else -np.power((state[index]) * 50, 2)

    def __call__(self, state):
        return self._height_penalty(state) + self._forward_tilt_penalty(state) + self._x_axis_angle_rotation_penalty(
            state)


class HumanoidVeryWideFlatTopQuadraticFi(Fi):
    def __call__(self, state):
        return 0.0 if 1.2 < state[0] < 1.6 else -np.power((1.4 - state[0]) * 100, 2)


class HumanoidEuclidean(Fi):
    def __call__(self, state):
        return - np.sqrt(np.power((1.4 - state[0]), 2)) * 1000


class FiFactory:
    FI_MAPPING = {
        'quadraticFlatTop': HumanoidFlatTopQuadraticFi,
        'quadraticNarrowFlatTop': HumanoidNarrowFlatTopQuadraticFi,
        'quadraticWideFlatTop': HumanoidWideFlatTopQuadraticFi,
        'quadraticWideFlatTopBigPenalty': HumanoidWideFlatTopQuadraticBigPenaltyFi,
        'quadraticWideFlatTopSmallPenalty': HumanoidWideFlatTopQuadraticSmallPenaltyFi,
        'quadraticVeryNarrowFlatTop': HumanoidVeryNarrowFlatTopQuadraticFi,
        'quadraticVeryWideFlatTop': HumanoidVeryWideFlatTopQuadraticFi,
        'quadraticWideFlatTopWithBodyTilt': HumanoidWideFlatTopQuadraticWithBodyTiltFi,
        'quadraticWideFlatTopWithBodyTiltLowerPenalty': HumanoidWideFlatTopQuadraticWithBodyTiltLowerPenaltyFi,
        'quadraticWideFlatTopWithForwardBodyTiltLowerPenalty': HumanoidWideFlatTopQuadraticWithForwardBodyTiltLowerPenaltyFi,
        'quadraticWideFlatTopWithBodyTiltWide': HumanoidWideFlatTopQuadraticWithBodyTiltWide
    }

    @staticmethod
    def get_fi(name: str):
        fi = FiFactory.FI_MAPPING.get(name, None)

        if fi is None:
            raise ValueError(f"Unknown fi: {name}, viable options are: {FiFactory.FI_MAPPING.keys()}")

        return fi()

    @staticmethod
    def register(name: str, _class=Type[Fi]):
        assert issubclass(_class, Fi), "Can only register classes that are subclasses of Fi"
        assert name not in FiFactory.FI_MAPPING, "This name is already taken"

        FiFactory.FI_MAPPING[name] = _class

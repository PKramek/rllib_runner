from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from src.constants import Constants
from src.reward_modifier.psi_base import Psi
from src.reward_modifier.util import normal_dist_density


def slightly_narrow_normal_dist_max_five(value: float, middle_of_dist: float):
    return 110 * normal_dist_density(value, middle_of_dist, 0.015) - 5.18


class AbstractHumanoidHeightTiltXAxis(Psi, ABC):

    def __call__(self, state: np.ndarray):
        return self._height_penalty(state) + self._forward_tilt_penalty(state) + self._x_axis_angle_rotation_penalty(
            state)

    @abstractmethod
    def _height_penalty(self, state: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _forward_tilt_penalty(self, state: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _x_axis_angle_rotation_penalty(self, state: np.ndarray):
        raise NotImplementedError


class AliveBonus(Psi):
    def __call__(self, state: np.ndarray) -> float:
        return 5.0


class AlivePenalty(Psi):
    def __call__(self, state: np.ndarray) -> float:
        return -5.0


# The best
class HeightPenaltySlightlyNarrow(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 110 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.015) - 5.18


# Class used to just test if it returns the same results as HeightPenaltySlightlyNarrow
class HeightTiltXAxisSlightlyNarrowJustHeight(AbstractHumanoidHeightTiltXAxis):
    def _height_penalty(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        return slightly_narrow_normal_dist_max_five(state[index], middle_of_dist)

    def _forward_tilt_penalty(self, state: np.ndarray):
        return 0.0

    def _x_axis_angle_rotation_penalty(self, state: np.ndarray):
        return 0.0


class HeightTiltXAxisSlightlyNarrow(AbstractHumanoidHeightTiltXAxis):
    def _height_penalty(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        return slightly_narrow_normal_dist_max_five(state[index], middle_of_dist)

    def _forward_tilt_penalty(self, state: np.ndarray):
        index = Constants.TILT_INDEX
        middle_of_dist = Constants.TILT_NOMINAL_VALUE

        return slightly_narrow_normal_dist_max_five(state[index], middle_of_dist)

    def _x_axis_angle_rotation_penalty(self, state: np.ndarray):
        index = Constants.X_AXIS_ROTATION_INDEX
        middle_of_dist = Constants.X_AXIS_ROTATION_NOMINAL_VALUE

        return slightly_narrow_normal_dist_max_five(state[index], middle_of_dist)


class HeightTiltXAxisSlightlyNarrowWeightedSixTwoTwo(HeightTiltXAxisSlightlyNarrow):
    def _height_penalty(self, state: np.ndarray):
        return super()._height_penalty(state) * 0.6

    def _forward_tilt_penalty(self, state: np.ndarray):
        return super()._forward_tilt_penalty(state) * 0.2

    def _x_axis_angle_rotation_penalty(self, state: np.ndarray):
        return super()._x_axis_angle_rotation_penalty(state) * 0.2


class HeightTiltXAxisSlightlyNarrowWeightedThreeThreeThree(HeightTiltXAxisSlightlyNarrow):
    def _height_penalty(self, state: np.ndarray):
        return super()._height_penalty(state) * 0.33

    def _forward_tilt_penalty(self, state: np.ndarray):
        return super()._forward_tilt_penalty(state) * 0.33

    def _x_axis_angle_rotation_penalty(self, state: np.ndarray):
        return super()._x_axis_angle_rotation_penalty(state) * 0.33


class HeightTiltXAxisSlightlyNarrowWeightedFiveThreeTwo(HeightTiltXAxisSlightlyNarrow):
    def _height_penalty(self, state: np.ndarray):
        return super()._height_penalty(state) * 0.5

    def _forward_tilt_penalty(self, state: np.ndarray):
        return super()._forward_tilt_penalty(state) * 0.3

    def _x_axis_angle_rotation_penalty(self, state: np.ndarray):
        return super()._x_axis_angle_rotation_penalty(state) * 0.32


class SquarePenalty(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        optimal_point = Constants.HEIGHT_NOMINAL_VALUE

        return - np.square((state[index] - optimal_point)) * 120


class PsiFactory:
    PSI_MAPPING = {
        'aliveBonus': AliveBonus,
        "alivePenalty": AlivePenalty,
        "heightSlightlyNarrowPenalty": HeightPenaltySlightlyNarrow,

        "heightTiltXAxisSlightlyNarrowPenalty": HeightTiltXAxisSlightlyNarrow,
        "heightTiltXAxisSlightlyNarrowPenaltyJustHeight": HeightTiltXAxisSlightlyNarrowJustHeight,
        "heightTiltXAxisSlightlyNarrowPenaltyWeightedSixTwoTwo": HeightTiltXAxisSlightlyNarrowWeightedSixTwoTwo,
        "heightTiltXAxisSlightlyNarrowPenaltyWeightedThreeThreeThree": HeightTiltXAxisSlightlyNarrowWeightedThreeThreeThree,
        "heightTiltXAxisSlightlyNarrowPenaltyWeightedFiveThreeTwo": HeightTiltXAxisSlightlyNarrowWeightedFiveThreeTwo,

        "heightSquarePenalty": SquarePenalty
    }

    @staticmethod
    def get_psi(name: str):
        fi = PsiFactory.PSI_MAPPING.get(name, None)

        if fi is None:
            raise ValueError(f"Unknown fi: {name}, viable options are: {PsiFactory.PSI_MAPPING.keys()}")

        return fi()

    @staticmethod
    def register(name: str, _class=Type[Psi]):
        assert issubclass(_class, Psi), "Can only register classes that are subclasses of Fi"
        assert name not in PsiFactory.PSI_MAPPING, "This name is already taken"

        PsiFactory.PSI_MAPPING[name] = _class

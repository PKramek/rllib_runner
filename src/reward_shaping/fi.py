from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from src.constants import Constants
from src.reward_shaping.fi_base import Fi
from src.reward_shaping.util import normal_dist_density


# This class is just an example and is not useful in any way
class SumFi(Fi):
    def __call__(self, state):
        return np.sum(state)


# Current Best
class HumanoidHeightNormalLowPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (10 * normal_dist_density(state[index], 1.4, 0.05)) + 500


################## HumanoidHeightNormalLowPenaltyShiftedExtremely with forward tilt

class AbstractHumanoidHeightTiltNormalLowPenaltyShifted(ABC):
    def __call__(self, state):
        return self._height_penalty(state) + self._forward_tilt_penalty(state) + self._x_axis_angle_rotation_penalty(
            state)

    @abstractmethod
    def _height_penalty(self, state):
        raise NotImplementedError

    @abstractmethod
    def _forward_tilt_penalty(self, state):
        raise NotImplementedError

    @abstractmethod
    def _x_axis_angle_rotation_penalty(self, state):
        raise NotImplementedError

    def _height_penalty_without_shift(self, state):
        index = Constants.HEIGHT_INDEX
        return 10 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.05)

    def _forward_tilt_penalty_without_shift(self, state):
        index = Constants.TILT_INDEX
        return 10 * normal_dist_density(state[index], Constants.TILT_NOMINAL_VALUE, 0.05)

    def _x_axis_angle_rotation_penalty_without_shift(self, state):
        index = Constants.X_AXIS_ROTATION_INDEX
        return 10 * normal_dist_density(state[index], Constants.X_AXIS_ROTATION_NOMINAL_VALUE, 0.05)


class HumanoidHeightTiltNormalLowPenaltyShiftedFiveNoneNone(AbstractHumanoidHeightTiltNormalLowPenaltyShifted):
    def _height_penalty(self, state):
        return self._height_penalty_without_shift(state) + 500

    def _forward_tilt_penalty(self, state):
        return 0

    def _x_axis_angle_rotation_penalty(self, state):
        return 0


class HumanoidHeightTiltNormalLowPenaltyShiftedFiveBaseBase(AbstractHumanoidHeightTiltNormalLowPenaltyShifted):
    def _height_penalty(self, state):
        return self._height_penalty_without_shift(state) + 500

    def _forward_tilt_penalty(self, state):
        return self._forward_tilt_penalty_without_shift(state)

    def _x_axis_angle_rotation_penalty(self, state):
        return self._x_axis_angle_rotation_penalty_without_shift(state)


class HumanoidHeightTiltNormalLowPenaltyShiftedThreeOneOne(AbstractHumanoidHeightTiltNormalLowPenaltyShifted):
    def _height_penalty(self, state):
        return self._height_penalty_without_shift(state) + 300

    def _forward_tilt_penalty(self, state):
        return self._forward_tilt_penalty_without_shift(state) + 100

    def _x_axis_angle_rotation_penalty(self, state):
        return self._x_axis_angle_rotation_penalty_without_shift(state) + 100


class HumanoidHeightTiltNormalLowPenaltyShiftedTwoTwoOne(AbstractHumanoidHeightTiltNormalLowPenaltyShifted):
    def _height_penalty(self, state):
        return self._height_penalty_without_shift(state) + 200.0

    def _forward_tilt_penalty(self, state):
        return self._forward_tilt_penalty_without_shift(state) + 200.0

    def _x_axis_angle_rotation_penalty(self, state):
        return self._x_axis_angle_rotation_penalty_without_shift(state) + 100.0


class HumanoidJustFive(AbstractHumanoidHeightTiltNormalLowPenaltyShifted):
    def _height_penalty(self, state):
        return 500.0

    def _forward_tilt_penalty(self, state):
        return 0.0

    def _x_axis_angle_rotation_penalty(self, state):
        return 0.0


############ New Normal
class HumanoidNewNormalSmallDiffMaxInFiveHundred():
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return 160 * normal_dist_density(state[index], 1.4, 1)


class HumanoidNewNormalBiggerDiffMaxInFiveHundred():
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (320 * normal_dist_density(state[index], 1.4, 0.5))


class HumanoidNewNormalBigDiffMaxInFiveHundred():
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (640 * normal_dist_density(state[index], 1.4, 0.25))


class FiFactory:
    FI_MAPPING = {
        'normalSmallSHiftedExtremally': HumanoidHeightNormalLowPenaltyShiftedExtremely,

        # Forward Tilt
        "normalHeightTiltSmallShiftedFiveNoneNone": HumanoidHeightTiltNormalLowPenaltyShiftedFiveNoneNone,
        "normalHeightTiltSmallShiftedFiveBaseBase": HumanoidHeightTiltNormalLowPenaltyShiftedFiveBaseBase,
        "normalHeightTiltSmallShiftedThreeOneOne": HumanoidHeightTiltNormalLowPenaltyShiftedThreeOneOne,
        "normalHeightTiltSmallShiftedTwoTwoOne": HumanoidHeightTiltNormalLowPenaltyShiftedTwoTwoOne,

        "justFiveHundred": HumanoidJustFive,

        # New Normal
        "newNormalSmallDiff": HumanoidNewNormalSmallDiffMaxInFiveHundred,
        "newNormalBiggerDiff": HumanoidNewNormalBiggerDiffMaxInFiveHundred,
        "newNormalBigDiff": HumanoidNewNormalBigDiffMaxInFiveHundred
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

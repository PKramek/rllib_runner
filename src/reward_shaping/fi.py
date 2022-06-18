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

class AbstractHumanoidMultipleDimensions(ABC):
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


class AbstractHumanoidHeightTiltNormalLowPenaltyShifted(AbstractHumanoidMultipleDimensions):
    def __call__(self, state):
        return self._height_penalty(state) + self._forward_tilt_penalty(state) + self._x_axis_angle_rotation_penalty(
            state)

    def _height_penalty_without_shift(self, state):
        index = Constants.HEIGHT_INDEX
        return 10 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.05)

    def _forward_tilt_penalty_without_shift(self, state):
        index = Constants.TILT_INDEX
        return 10 * normal_dist_density(state[index], Constants.TILT_NOMINAL_VALUE, 0.05)

    def _x_axis_angle_rotation_penalty_without_shift(self, state):
        index = Constants.X_AXIS_ROTATION_INDEX
        return 10 * normal_dist_density(state[index], Constants.X_AXIS_ROTATION_NOMINAL_VALUE, 0.05)


class HumanoidJustFive(AbstractHumanoidMultipleDimensions):
    def _height_penalty(self, state):
        return 500.0

    def _forward_tilt_penalty(self, state):
        return 0.0

    def _x_axis_angle_rotation_penalty(self, state):
        return 0.0


def func_normal_narrow(value: float, middle_of_normal_dist: float):
    return 3200 * normal_dist_density(value, middle_of_normal_dist, 0.05)


def func_normal_super_narrow(value: float, middle_of_normal_dist: float):
    return 6400 * normal_dist_density(value, middle_of_normal_dist, 0.02)


# Just body height heuristic

class HumanoidNewNormalNarrow():
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE
        return func_normal_narrow(state[index], middle_of_dist)


class HumanoidNewNormalNarrowSmallDiff(HumanoidNewNormalNarrow):
    def __call__(self, state):
        return super().__call__(state) / 2


class HumanoidNewNormalNarrowBigDiff(HumanoidNewNormalNarrow):
    def __call__(self, state):
        return super().__call__(state) * 2


class HumanoidNewNormalSuperNarrow():
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE
        return func_normal_super_narrow(state[index], middle_of_dist)


class HumanoidNewNormalSuperNarrowSmallDiff(HumanoidNewNormalSuperNarrow):
    def __call__(self, state):
        return super().__call__(state) / 2


class HumanoidNewNormalSuperNarrowBigDiff(HumanoidNewNormalSuperNarrow):
    def __call__(self, state):
        return super().__call__(state) * 2


# Both heuristics

class HumanoidBothHeuristicsNewNormalNarrow(AbstractHumanoidMultipleDimensions):
    def _height_penalty(self, state):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        return func_normal_narrow(state[index], middle_of_dist)

    def _forward_tilt_penalty(self, state):
        index = Constants.TILT_INDEX
        middle_of_dist = Constants.TILT_NOMINAL_VALUE

        return func_normal_narrow(state[index], middle_of_dist)

    def _x_axis_angle_rotation_penalty(self, state):
        index = Constants.X_AXIS_ROTATION_INDEX
        middle_of_dist = Constants.X_AXIS_ROTATION_NOMINAL_VALUE

        return func_normal_narrow(state[index], middle_of_dist)


class HumanoidBothHeuristicsNewNormalNarrowBigDiff(HumanoidBothHeuristicsNewNormalNarrow):
    def _height_penalty(self, state):
        return super()._height_penalty(state) * 2

    def _forward_tilt_penalty(self, state):
        return super()._forward_tilt_penalty(state) * 2

    def _x_axis_angle_rotation_penalty(self, state):
        return super()._x_axis_angle_rotation_penalty(state) * 2


class HumanoidBothHeuristicsNewNormalNarrowSmallDiff(HumanoidBothHeuristicsNewNormalNarrow):
    def _height_penalty(self, state):
        return super()._height_penalty(state) / 2

    def _forward_tilt_penalty(self, state):
        return super()._forward_tilt_penalty(state) / 2

    def _x_axis_angle_rotation_penalty(self, state):
        return super()._x_axis_angle_rotation_penalty(state) / 2


class HumanoidBothHeuristicsNewNormalSuperNarrow(AbstractHumanoidMultipleDimensions):
    def _height_penalty(self, state):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        return func_normal_super_narrow(state[index], middle_of_dist)

    def _forward_tilt_penalty(self, state):
        index = Constants.TILT_INDEX
        middle_of_dist = Constants.TILT_NOMINAL_VALUE

        return func_normal_super_narrow(state[index], middle_of_dist)

    def _x_axis_angle_rotation_penalty(self, state):
        index = Constants.X_AXIS_ROTATION_INDEX
        middle_of_dist = Constants.X_AXIS_ROTATION_NOMINAL_VALUE

        return func_normal_super_narrow(state[index], middle_of_dist)


class HumanoidBothHeuristicsNewNormalSuperNarrowBigDiff(HumanoidBothHeuristicsNewNormalSuperNarrow):
    def _height_penalty(self, state):
        return super()._height_penalty(state) * 2

    def _forward_tilt_penalty(self, state):
        return super()._forward_tilt_penalty(state) * 2

    def _x_axis_angle_rotation_penalty(self, state):
        return super()._x_axis_angle_rotation_penalty(state) * 2


class HumanoidBothHeuristicsNewNormalSuperNarrowSmallDiff(HumanoidBothHeuristicsNewNormalSuperNarrow):
    def _height_penalty(self, state):
        return super()._height_penalty(state) / 2

    def _forward_tilt_penalty(self, state):
        return super()._forward_tilt_penalty(state) / 2

    def _x_axis_angle_rotation_penalty(self, state):
        return super()._x_axis_angle_rotation_penalty(state) / 2


class FiFactory:
    FI_MAPPING = {
        'normalSmallSHiftedExtremally': HumanoidHeightNormalLowPenaltyShiftedExtremely,

        "justFiveHundred": HumanoidJustFive,

        # New Normal
        "newNormalNarrow": HumanoidNewNormalNarrow,
        "newNormalNarrowBigDiff": HumanoidNewNormalNarrowBigDiff,
        "newNormalNarrowSmallDiff": HumanoidNewNormalNarrowSmallDiff,

        # New Normal Narrow
        "newNormalSuperNarrow": HumanoidNewNormalSuperNarrow,
        "newNormalSuperNarrowBigDiff": HumanoidNewNormalSuperNarrow,
        "newNormalSuperNarrowSmallDiff": HumanoidNewNormalSuperNarrow,

        # New Normal Both Heuristics
        "newNormalNarrowBothHeuristics": HumanoidBothHeuristicsNewNormalNarrow,
        "newNormalNarrowBothHeuristicsBigDiff": HumanoidBothHeuristicsNewNormalNarrowBigDiff,
        "newNormalNarrowBothHeuristicsSmallDiff": HumanoidBothHeuristicsNewNormalNarrowSmallDiff,

        # New Normal Narrow Both Heuristics
        "newNormalSuperNarrowBothHeuristics": HumanoidBothHeuristicsNewNormalSuperNarrow,
        "newNormalSuperNarrowBothHeuristicsBigDiff": HumanoidBothHeuristicsNewNormalSuperNarrowBigDiff,
        "newNormalSuperNarrowBothHeuristicsSmallDiff": HumanoidBothHeuristicsNewNormalSuperNarrowSmallDiff,
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

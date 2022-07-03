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


# Current Best
class HumanoidHeightNormalLowPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (10 * normal_dist_density(state[index], 1.4, 0.05)) + 500


class HumanoidJustFive(AbstractHumanoidMultipleDimensions):
    def _height_penalty(self, state):
        return 500.0

    def _forward_tilt_penalty(self, state):
        return 0.0

    def _x_axis_angle_rotation_penalty(self, state):
        return 0.0


################################

class HeightPenaltySlightlyNarrow(Fi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 110 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.015) - 5.18


class HeightPenaltySlightlyNarrowFive(HeightPenaltySlightlyNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 5


class HeightPenaltySlightlyNarrowTen(HeightPenaltySlightlyNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 10


class HeightPenaltySlightlyNarrowTwenty(HeightPenaltySlightlyNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 20


class HeightPenaltySlightlyNarrowFifty(HeightPenaltySlightlyNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 50


class HeightPenaltySlightlyNarrowSeventy(HeightPenaltySlightlyNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 70


class HeightPenaltySlightlyNarrowHundred(HeightPenaltySlightlyNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 100


class HeightPenaltySlightlyNarrowHundredFifty(HeightPenaltySlightlyNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 150


class HeightPenaltySlightlyNarrowHundredTwoHundred(HeightPenaltySlightlyNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 200


#################################################################################3
class HeightPenaltyLessNarrow(Fi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 55 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.03) - 5.18


class HeightPenaltyLessNarrowFive(HeightPenaltyLessNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 5


class HeightPenaltyLessNarrowTen(HeightPenaltyLessNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 10


class HeightPenaltyLessNarrowTwenty(HeightPenaltyLessNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 20


class HeightPenaltyLessNarrowFifty(HeightPenaltyLessNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 50


class HeightPenaltyLessNarrowSeventy(HeightPenaltyLessNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 70


class HeightPenaltyLessNarrowHundred(HeightPenaltyLessNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 100


class HeightPenaltyLessNarrowHundredFifty(HeightPenaltyLessNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 150


class HeightPenaltyLessNarrowTwoHundred(HeightPenaltyLessNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 200


#####################
class SquarePenalty(Fi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        optimal_point = Constants.HEIGHT_NOMINAL_VALUE

        return - np.square((state[index] - optimal_point)) * 120


class SquarePenaltyTen(SquarePenalty):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 10


class SquarePenaltyTwenty(SquarePenalty):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 20


class SquarePenaltyFifty(SquarePenalty):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 50


class SquarePenaltySeventy(SquarePenalty):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 70


class SquarePenaltyHundred(SquarePenalty):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 100


class SquarePenaltyHundredFifty(SquarePenalty):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 150


class SquarePenaltyTwoHundred(SquarePenalty):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 200


class SquarePenaltyFourHundred(SquarePenalty):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 400


class FiFactory:
    FI_MAPPING = {
        'normalSmallSHiftedExtremally': HumanoidHeightNormalLowPenaltyShiftedExtremely,
        "justFiveHundred": HumanoidJustFive,

        "heightSlightlyNarrowPenalty": HeightPenaltySlightlyNarrow,
        "heightSlightlyNarrowPenaltyTen": HeightPenaltySlightlyNarrowTen,
        "heightSlightlyNarrowPenaltyTwenty": HeightPenaltySlightlyNarrowTwenty,
        "heightSlightlyNarrowPenaltyFifty": HeightPenaltySlightlyNarrowFifty,
        "heightSlightlyNarrowPenaltySeventy": HeightPenaltySlightlyNarrowSeventy,
        "heightSlightlyNarrowPenaltyHundred": HeightPenaltySlightlyNarrowHundred,
        "heightSlightlyNarrowPenaltyHundredFifty": HeightPenaltySlightlyNarrowHundredFifty,
        "heightSlightlyNarrowPenaltyTwoHundred": HeightPenaltySlightlyNarrowHundredTwoHundred,

        "heightLessNarrowPenalty": HeightPenaltyLessNarrow,
        "heightLessNarrowPenaltyTen": HeightPenaltyLessNarrowTen,
        "heightLessNarrowPenaltyTwenty": HeightPenaltyLessNarrowTwenty,
        "heightLessNarrowPenaltyFifty": HeightPenaltyLessNarrowFifty,
        "heightLessNarrowPenaltySeventy": HeightPenaltyLessNarrowSeventy,
        "heightLessNarrowPenaltyHundred": HeightPenaltyLessNarrowHundred,
        "heightLessNarrowPenaltyHundredFifty": HeightPenaltyLessNarrowHundredFifty,
        "heightLessNarrowPenaltyTwoHundred": HeightPenaltyLessNarrowTwoHundred,

        "heightSquarePenalty": SquarePenalty,
        "heightSquarePenaltyTen": SquarePenaltyTen,
        "heightSquarePenaltyTwenty": SquarePenaltyTwenty,
        "heightSquarePenaltyFifty": SquarePenaltyFifty,
        "heightSquarePenaltySeventy": SquarePenaltySeventy,

        "heightSquarePenaltyHundred": SquarePenaltyHundred,
        "heightSquarePenaltyHundredFifty": SquarePenaltyHundredFifty,
        "heightSquarePenaltyTwoHundred": SquarePenaltyTwoHundred,
        "heightSquarePenaltyFourHundred": SquarePenaltyFourHundred,

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

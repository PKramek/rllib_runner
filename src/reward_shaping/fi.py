from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from scipy.stats import t

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


######################

class HeightPenaltyLessNarrowFlipped(HeightPenaltyLessNarrow):
    def __call__(self, state: np.ndarray) -> float:
        return - super().__call__(state)


class HeightPenaltyLessNarrowFiveFlipped(HeightPenaltyLessNarrowFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 5


class HeightPenaltyLessNarrowTenFlipped(HeightPenaltyLessNarrowFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 10


class HeightPenaltyLessNarrowTwentyFlipped(HeightPenaltyLessNarrowFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 20


class HeightPenaltyLessNarrowFiftyFlipped(HeightPenaltyLessNarrowFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 50


class HeightPenaltyLessNarrowSeventyFlipped(HeightPenaltyLessNarrowFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 70


class HeightPenaltyLessNarrowHundredFlipped(HeightPenaltyLessNarrowFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 100


class HeightPenaltyLessNarrowHundredFiftyFlipped(HeightPenaltyLessNarrowFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 150


class HeightPenaltyLessNarrowTwoHundredFlipped(HeightPenaltyLessNarrowFlipped):
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


#########################3
class SquarePenaltyFlipped(SquarePenalty):
    def __call__(self, state: np.ndarray) -> float:
        return - super().__call__(state)


class SquarePenaltyTenFlipped(SquarePenaltyFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 10


class SquarePenaltyTwentyFlipped(SquarePenaltyFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 20


class SquarePenaltyFiftyFlipped(SquarePenaltyFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 50


class SquarePenaltySeventyFlipped(SquarePenaltyFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 70


class SquarePenaltyHundredFlipped(SquarePenaltyFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 100


class SquarePenaltyHundredFiftyFlipped(SquarePenaltyFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 150


class SquarePenaltyTwoHundredFlipped(SquarePenaltyFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 200


class SquarePenaltyFourHundredFlipped(SquarePenaltyFlipped):
    def __call__(self, state: np.ndarray) -> float:
        return super().__call__(state) * 400


######################################
class TStudentHeightNormal(Fi):

    def __call__(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        degree_of_freedom = 0.5
        scale = 0.05

        return t.pdf(state[index], df=degree_of_freedom, scale=scale, loc=middle_of_dist)


class TStudentHeightNormalTen(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 10


class TStudentHeightNormalTwenty(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 20


class TStudentHeightNormalFifty(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 50


class TStudentHeightNormalSeventy(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 70


class TStudentHeightNormalHundred(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 100


class TStudentHeightNormalHundredFifty(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 150


class TStudentHeightNormalTwoHundred(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 200


##

class TStudentHeightNormalFlipped(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return -super().__call__(state)


class TStudentHeightNormalTenFlipped(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return -super().__call__(state) * 10


class TStudentHeightNormalTwentyFlipped(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 20


class TStudentHeightNormalFiftyFlipped(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 50


class TStudentHeightNormalSeventyFlipped(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 70


class TStudentHeightNormalHundredFlipped(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 100


class TStudentHeightNormalHundredFiftyFlipped(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 150


class TStudentHeightNormalTwoHundredFlipped(TStudentHeightNormal):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 200


######################################
class TStudentHeightWide(Fi):

    def __call__(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        degree_of_freedom = 5.0
        scale = 0.07

        return t.pdf(state[index], df=degree_of_freedom, scale=scale, loc=middle_of_dist)


class TStudentHeightWideTen(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 10


class TStudentHeightWideTwenty(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 20


class TStudentHeightWideFifty(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 50


class TStudentHeightWideSeventy(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 70


class TStudentHeightWideHundred(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 100


class TStudentHeightWideHundredFifty(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 150


class TStudentHeightWideTwoHundred(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 200


class TStudentHeightWideFlipped(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return -super().__call__(state)


class TStudentHeightWideTenFlipped(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return -super().__call__(state) * 10


class TStudentHeightWideTwentyFlipped(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 20


class TStudentHeightWideFiftyFlipped(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 50


class TStudentHeightWideSeventyFlipped(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 70


class TStudentHeightWideHundredFlipped(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 100


class TStudentHeightWideHundredFiftyFlipped(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 150


class TStudentHeightWideTwoHundredFlipped(TStudentHeightWide):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 200


#####################3


class TStudentHeightNarrow(Fi):

    def __call__(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        degree_of_freedom = 0.1
        scale = 0.03

        return t.pdf(state[index], df=degree_of_freedom, scale=scale, loc=middle_of_dist)


class TStudentHeightNarrowTen(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 10


class TStudentHeightNarrowTwenty(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 20


class TStudentHeightNarrowFifty(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 50


class TStudentHeightNarrowSeventy(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 70


class TStudentHeightNarrowHundred(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 100


class TStudentHeightNarrowHundredFifty(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 150


class TStudentHeightNarrowTwoHundred(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return super().__call__(state) * 200


class TStudentHeightNarrowFlipped(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return -super().__call__(state)


class TStudentHeightNarrowTenFlipped(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return -super().__call__(state) * 10


class TStudentHeightNarrowTwentyFlipped(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 20


class TStudentHeightNarrowFiftyFlipped(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 50


class TStudentHeightNarrowSeventyFlipped(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 70


class TStudentHeightNarrowHundredFlipped(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 100


class TStudentHeightNarrowHundredFiftyFlipped(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 150


class TStudentHeightNarrowTwoHundredFlipped(TStudentHeightNarrow):

    def __call__(self, state: np.ndarray):
        return - super().__call__(state) * 200


#####################################


class FiFactory:
    FI_MAPPING = {
        'normalSmallSHiftedExtremally': HumanoidHeightNormalLowPenaltyShiftedExtremely,
        "justFiveHundred": HumanoidJustFive,

        "tstudentHeightNormal": TStudentHeightNormal,
        "tstudentHeightNormalTen": TStudentHeightNormalTen,
        "tstudentHeightNormalTwenty": TStudentHeightNormalTwenty,
        "tstudentHeightNormalFifty": TStudentHeightNormalFifty,
        "tstudentHeightNormalSeventy": TStudentHeightNormalSeventy,
        "tstudentHeightNormalHundred": TStudentHeightNormalHundred,
        "tstudentHeightNormalHundredFifty": TStudentHeightNormalHundredFifty,
        "tstudentHeightNormalTwoHundred": TStudentHeightNormalTwoHundred,

        "tstudentHeightNormalFlipped": TStudentHeightNormalFlipped,
        "tstudentHeightNormalTenFlipped": TStudentHeightNormalTenFlipped,
        "tstudentHeightNormalTwentyFlipped": TStudentHeightNormalTwentyFlipped,
        "tstudentHeightNormalFiftyFlipped": TStudentHeightNormalFiftyFlipped,
        "tstudentHeightNormalSeventyFlipped": TStudentHeightNormalSeventyFlipped,
        "tstudentHeightNormalHundredFlipped": TStudentHeightNormalHundredFlipped,
        "tstudentHeightNormalHundredFiftyFlipped": TStudentHeightNormalHundredFiftyFlipped,
        "tstudentHeightNormalTwoHundredFlipped": TStudentHeightNormalTwoHundredFlipped,

        "tstudentHeightWide": TStudentHeightWide,
        "tstudentHeightWideTen": TStudentHeightWideTen,
        "tstudentHeightWideTwenty": TStudentHeightWideTwenty,
        "tstudentHeightWideFifty": TStudentHeightWideFifty,
        "tstudentHeightWideSeventy": TStudentHeightWideSeventy,
        "tstudentHeightWideHundred": TStudentHeightWideHundred,
        "tstudentHeightWideHundredFifty": TStudentHeightWideHundredFifty,
        "tstudentHeightWideTwoHundred": TStudentHeightWideTwoHundred,

        "tstudentHeightWideFlipped": TStudentHeightWideFlipped,
        "tstudentHeightWideTenFlipped": TStudentHeightWideTenFlipped,
        "tstudentHeightWideTwentyFlipped": TStudentHeightWideTwentyFlipped,
        "tstudentHeightWideFiftyFlipped": TStudentHeightWideFiftyFlipped,
        "tstudentHeightWideSeventyFlipped": TStudentHeightWideSeventyFlipped,
        "tstudentHeightWideHundredFlipped": TStudentHeightWideHundredFlipped,
        "tstudentHeightWideHundredFiftyFlipped": TStudentHeightWideHundredFiftyFlipped,
        "tstudentHeightWideTwoHundredFlipped": TStudentHeightWideTwoHundredFlipped,

        "tstudentHeightNarrow": TStudentHeightNarrow,
        "tstudentHeightNarrowTen": TStudentHeightNarrowTen,
        "tstudentHeightNarrowTwenty": TStudentHeightNarrowTwenty,
        "tstudentHeightNarrowFifty": TStudentHeightNarrowFifty,
        "tstudentHeightNarrowSeventy": TStudentHeightNarrowSeventy,
        "tstudentHeightNarrowHundred": TStudentHeightNarrowHundred,
        "tstudentHeightNarrowHundredFifty": TStudentHeightNarrowHundredFifty,
        "tstudentHeightNarrowTwoHundred": TStudentHeightNarrowTwoHundred,

        "tstudentHeightNarrowFlipped": TStudentHeightNarrowFlipped,
        "tstudentHeightNarrowTenFlipped": TStudentHeightNarrowTenFlipped,
        "tstudentHeightNarrowTwentyFlipped": TStudentHeightNarrowTwentyFlipped,
        "tstudentHeightNarrowFiftyFlipped": TStudentHeightNarrowFiftyFlipped,
        "tstudentHeightNarrowSeventyFlipped": TStudentHeightNarrowSeventyFlipped,
        "tstudentHeightNarrowHundredFlipped": TStudentHeightNarrowHundredFlipped,
        "tstudentHeightNarrowHundredFiftyFlipped": TStudentHeightNarrowHundredFiftyFlipped,
        "tstudentHeightNarrowTwoHundredFlipped": TStudentHeightNarrowTwoHundredFlipped,
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

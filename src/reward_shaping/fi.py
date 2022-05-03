from typing import Type

import numpy as np

import normal_shifted_four_hundred
import normal_shifted_one_hundred
import normal_shifted_seven_hundred
from src.constants import Constants
from src.reward_shaping.fi_base import Fi
from src.reward_shaping.util import normal_dist_density


# This class is just an example and is not useful in any way
class SumFi(Fi):
    def __call__(self, state):
        return np.sum(state)


class HumanoidHeightLinearNotFlat(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return -np.abs((1.4 - state[index]) * 10)


class HumanoidHeightLinearLowerPenaltyNotFlat(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return -np.abs((1.4 - state[index]) * 5)


class HumanoidHeightLinearHigherPenaltyNotFlat(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return -np.abs((1.4 - state[index]) * 50)


class HumanoidHeightLinearShiftedUpNotFlat(HumanoidHeightLinearNotFlat):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi + 5


class HumanoidHeightLinearShiftedUpSlightlyNotFlat(HumanoidHeightLinearNotFlat):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi + 1


class HumanoidHeightLinearShiftedUpMassivelyNotFlat(HumanoidHeightLinearNotFlat):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi + 20


class HumanoidHeightLinearShiftedUpExcessivelyNotFlat(HumanoidHeightLinearNotFlat):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi + 300


class HumanoidHeightLinearShiftedUpExtremelyNotFlat(HumanoidHeightLinearNotFlat):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi + 500


############################ SMALL ############################
class HumanoidHeightNormalSmallPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (5 * normal_dist_density(state[index], 1.4, 0.05)) + 500


class HumanoidHeightNormalNarrowSmallPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (25 * normal_dist_density(state[index], 1.4, 0.01)) + 500


class HumanoidHeightNormalWideSmallPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (2.5 * normal_dist_density(state[index], 1.4, 0.1)) + 500


############################ LOW  ############################
class HumanoidHeightNormalLowPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (10 * normal_dist_density(state[index], 1.4, 0.05)) + 500


class HumanoidHeightNormalNarrowLowPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (50 * normal_dist_density(state[index], 1.4, 0.01)) + 500


class HumanoidHeightNormalWideLowPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (5 * normal_dist_density(state[index], 1.4, 0.1)) + 500


############################ STANDARD ############################

class HumanoidHeightNormalStandardPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (30 * normal_dist_density(state[index], 1.4, 0.05)) + 500


class HumanoidHeightNormalNarrowStandardPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (150 * normal_dist_density(state[index], 1.4, 0.01)) + 500


class HumanoidHeightNormalWideStandardPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (15 * normal_dist_density(state[index], 1.4, 0.1)) + 500


############################ HIGH ############################


class HumanoidHeightNormalHighPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (60 * normal_dist_density(state[index], 1.4, 0.05)) + 500


class HumanoidHeightNormalNarrowHighPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (300 * normal_dist_density(state[index], 1.4, 0.01)) + 500


class HumanoidHeightNormalWideHighPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (30 * normal_dist_density(state[index], 1.4, 0.1)) + 500


class FiFactory:
    FI_MAPPING = {
        'linearNotFlat': HumanoidHeightLinearNotFlat,
        'linearLowerPenaltyNotFlat': HumanoidHeightLinearLowerPenaltyNotFlat,
        'linearHigherPenaltyNotFlat': HumanoidHeightLinearHigherPenaltyNotFlat,
        'linearShiftedUpNotFlat': HumanoidHeightLinearShiftedUpNotFlat,
        'linearShiftedUpSlightlyNotFlat': HumanoidHeightLinearShiftedUpSlightlyNotFlat,
        'linearShiftedUpMassivelyNotFlat': HumanoidHeightLinearShiftedUpMassivelyNotFlat,
        'linearShiftedUpExcessivelyNotFlat': HumanoidHeightLinearShiftedUpExcessivelyNotFlat,
        'linearShiftedUpExtremallyNotFlat': HumanoidHeightLinearShiftedUpExtremelyNotFlat,

        'normalSmallSHiftedExtremally': HumanoidHeightNormalLowPenaltyShiftedExtremely,  # left here for db reference

        'normalSmallShiftedUpExtremely': HumanoidHeightNormalSmallPenaltyShiftedExtremely,
        'normalSmallWideShiftedUpExtremely': HumanoidHeightNormalWideSmallPenaltyShiftedExtremely,
        'normalSmallNarrowShiftedUpExtremely': HumanoidHeightNormalNarrowSmallPenaltyShiftedExtremely,

        'normalLowShiftedUpExtremely': HumanoidHeightNormalLowPenaltyShiftedExtremely,
        'normalLowWideShiftedUpExtremely': HumanoidHeightNormalWideLowPenaltyShiftedExtremely,
        'normalLowNarrowShiftedUpExtremely': HumanoidHeightNormalNarrowLowPenaltyShiftedExtremely,

        'normalStandardShiftedUpExtremely': HumanoidHeightNormalStandardPenaltyShiftedExtremely,
        'normalStandardWideShiftedUpExtremely': HumanoidHeightNormalWideStandardPenaltyShiftedExtremely,
        'normalStandardNarrowShiftedUpExtremely': HumanoidHeightNormalNarrowStandardPenaltyShiftedExtremely,

        'normalHighShiftedUpExtremely': HumanoidHeightNormalHighPenaltyShiftedExtremely,
        'normalHighWideShiftedUpExtremely': HumanoidHeightNormalWideHighPenaltyShiftedExtremely,
        'normalHighNarrowShiftedUpExtremely': HumanoidHeightNormalNarrowHighPenaltyShiftedExtremely,

        ################################################################################################################
        'normalSmallShiftedUpFourHundred': normal_shifted_four_hundred.HumanoidHeightNormalSmallPenaltyShiftedFourHundred,
        'normalSmallWideShiftedUpFourHundred': normal_shifted_four_hundred.HumanoidHeightNormalWideSmallPenaltyShiftedFourHundred,
        'normalSmallNarrowShiftedUpFourHundred': normal_shifted_four_hundred.HumanoidHeightNormalNarrowSmallPenaltyShiftedFourHundred,

        'normalLowShiftedUpFourHundred': normal_shifted_four_hundred.HumanoidHeightNormalLowPenaltyShiftedFourHundred,
        'normalLowWideShiftedUpFourHundred': normal_shifted_four_hundred.HumanoidHeightNormalWideLowPenaltyShiftedFourHundred,
        'normalLowNarrowShiftedUpFourHundred': normal_shifted_four_hundred.HumanoidHeightNormalNarrowLowPenaltyShiftedFourHundred,

        'normalStandardShiftedUpFourHundred': normal_shifted_four_hundred.HumanoidHeightNormalStandardPenaltyShiftedFourHundred,
        'normalStandardWideShiftedUpFourHundred': normal_shifted_four_hundred.HumanoidHeightNormalWideStandardPenaltyShiftedFourHundred,
        'normalStandardNarrowShiftedUpFourHundred': normal_shifted_four_hundred.HumanoidHeightNormalNarrowStandardPenaltyShiftedFourHundred,

        'normalHighShiftedUpFourHundred': normal_shifted_four_hundred.HumanoidHeightNormalHighPenaltyShiftedFourHundred,
        'normalHighWideShiftedUpFourHundred': normal_shifted_four_hundred.HumanoidHeightNormalWideHighPenaltyShiftedFourHundred,
        'normalHighNarrowShiftedUpFourHundred': normal_shifted_four_hundred.HumanoidHeightNormalNarrowHighPenaltyShiftedFourHundred,

        ################################################################################################################
        'normalSmallShiftedUpSevenHundred': normal_shifted_seven_hundred.HumanoidHeightNormalSmallPenaltyShiftedSevenHundred,
        'normalSmallWideShiftedUpSevenHundred': normal_shifted_seven_hundred.HumanoidHeightNormalWideSmallPenaltyShiftedSevenHundred,
        'normalSmallNarrowShiftedUpSevenHundred': normal_shifted_seven_hundred.HumanoidHeightNormalNarrowSmallPenaltyShiftedSevenHundred,

        'normalLowShiftedUpSevenHundred': normal_shifted_seven_hundred.HumanoidHeightNormalLowPenaltyShiftedSevenHundred,
        'normalLowWideShiftedUpSevenHundred': normal_shifted_seven_hundred.HumanoidHeightNormalWideLowPenaltyShiftedSevenHundred,
        'normalLowNarrowShiftedUpSevenHundred': normal_shifted_seven_hundred.HumanoidHeightNormalNarrowLowPenaltyShiftedSevenHundred,

        'normalStandardShiftedUpSevenHundred': normal_shifted_seven_hundred.HumanoidHeightNormalStandardPenaltyShiftedSevenHundred,
        'normalStandardWideShiftedUpSevenHundred': normal_shifted_seven_hundred.HumanoidHeightNormalWideStandardPenaltyShiftedSevenHundred,
        'normalStandardNarrowShiftedUpSevenHundred': normal_shifted_seven_hundred.HumanoidHeightNormalNarrowStandardPenaltyShiftedSevenHundred,

        'normalHighShiftedUpSevenHundred': normal_shifted_seven_hundred.HumanoidHeightNormalHighPenaltyShiftedSevenHundred,
        'normalHighWideShiftedUpSevenHundred': normal_shifted_seven_hundred.HumanoidHeightNormalWideHighPenaltyShiftedSevenHundred,
        'normalHighNarrowShiftedUpSevenHundred': normal_shifted_seven_hundred.HumanoidHeightNormalNarrowHighPenaltyShiftedSevenHundred,

        ################################################################################################################

        'normalSmallShiftedUpOneHundred': normal_shifted_one_hundred.HumanoidHeightNormalSmallPenaltyShiftedOneHundred,
        'normalSmallWideShiftedUpOneHundred': normal_shifted_one_hundred.HumanoidHeightNormalWideSmallPenaltyShiftedOneHundred,
        'normalSmallNarrowShiftedUpOneHundred': normal_shifted_one_hundred.HumanoidHeightNormalNarrowSmallPenaltyShiftedOneHundred,

        'normalLowShiftedUpOneHundred': normal_shifted_one_hundred.HumanoidHeightNormalLowPenaltyShiftedOneHundred,
        'normalLowWideShiftedUpOneHundred': normal_shifted_one_hundred.HumanoidHeightNormalWideLowPenaltyShiftedOneHundred,
        'normalLowNarrowShiftedUpOneHundred': normal_shifted_one_hundred.HumanoidHeightNormalNarrowLowPenaltyShiftedOneHundred,

        'normalStandardShiftedUpOneHundred': normal_shifted_one_hundred.HumanoidHeightNormalStandardPenaltyShiftedOneHundred,
        'normalStandardWideShiftedUpOneHundred': normal_shifted_one_hundred.HumanoidHeightNormalWideStandardPenaltyShiftedOneHundred,
        'normalStandardNarrowShiftedUpOneHundred': normal_shifted_one_hundred.HumanoidHeightNormalNarrowStandardPenaltyShiftedOneHundred,

        'normalHighShiftedUpOneHundred': normal_shifted_one_hundred.HumanoidHeightNormalHighPenaltyShiftedOneHundred,
        'normalHighWideShiftedUpOneHundred': normal_shifted_one_hundred.HumanoidHeightNormalWideHighPenaltyShiftedOneHundred,
        'normalHighNarrowShiftedUpOneHundred': normal_shifted_one_hundred.HumanoidHeightNormalNarrowHighPenaltyShiftedOneHundred,

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

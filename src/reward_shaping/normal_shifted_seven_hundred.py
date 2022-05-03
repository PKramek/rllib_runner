from src.constants import Constants
from src.reward_shaping.fi_base import Fi
from src.reward_shaping.util import normal_dist_density

SEVEN_HUNDRED = 700


############################ SMALL ############################

class HumanoidHeightNormalSmallPenaltyShiftedSevenHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (5 * normal_dist_density(state[index], 1.4, 0.05)) + SEVEN_HUNDRED


class HumanoidHeightNormalNarrowSmallPenaltyShiftedSevenHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (25 * normal_dist_density(state[index], 1.4, 0.01)) + SEVEN_HUNDRED


class HumanoidHeightNormalWideSmallPenaltyShiftedSevenHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (2.5 * normal_dist_density(state[index], 1.4, 0.1)) + SEVEN_HUNDRED


############################ LOW  ############################

class HumanoidHeightNormalLowPenaltyShiftedSevenHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (10 * normal_dist_density(state[index], 1.4, 0.05)) + SEVEN_HUNDRED


class HumanoidHeightNormalNarrowLowPenaltyShiftedSevenHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (50 * normal_dist_density(state[index], 1.4, 0.01)) + SEVEN_HUNDRED


class HumanoidHeightNormalWideLowPenaltyShiftedSevenHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (5 * normal_dist_density(state[index], 1.4, 0.1)) + SEVEN_HUNDRED


############################ STANDARD ############################

class HumanoidHeightNormalStandardPenaltyShiftedSevenHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (30 * normal_dist_density(state[index], 1.4, 0.05)) + SEVEN_HUNDRED


class HumanoidHeightNormalNarrowStandardPenaltyShiftedSevenHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (150 * normal_dist_density(state[index], 1.4, 0.01)) + SEVEN_HUNDRED


class HumanoidHeightNormalWideStandardPenaltyShiftedSevenHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (15 * normal_dist_density(state[index], 1.4, 0.1)) + SEVEN_HUNDRED


############################ HIGH ############################


class HumanoidHeightNormalHighPenaltyShiftedSevenHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (60 * normal_dist_density(state[index], 1.4, 0.05)) + SEVEN_HUNDRED


class HumanoidHeightNormalNarrowHighPenaltyShiftedSevenHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (300 * normal_dist_density(state[index], 1.4, 0.01)) + SEVEN_HUNDRED


class HumanoidHeightNormalWideHighPenaltyShiftedSevenHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (30 * normal_dist_density(state[index], 1.4, 0.1)) + SEVEN_HUNDRED

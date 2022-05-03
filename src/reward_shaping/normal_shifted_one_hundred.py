from src.constants import Constants
from src.reward_shaping.fi_base import Fi
from src.reward_shaping.util import normal_dist_density

ONE_HUNDRED = 100


############################ SMALL ############################

class HumanoidHeightNormalSmallPenaltyShiftedOneHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (5 * normal_dist_density(state[index], 1.4, 0.05)) + ONE_HUNDRED


class HumanoidHeightNormalNarrowSmallPenaltyShiftedOneHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (25 * normal_dist_density(state[index], 1.4, 0.01)) + ONE_HUNDRED


class HumanoidHeightNormalWideSmallPenaltyShiftedOneHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (2.5 * normal_dist_density(state[index], 1.4, 0.1)) + ONE_HUNDRED


############################ LOW  ############################

class HumanoidHeightNormalLowPenaltyShiftedOneHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (10 * normal_dist_density(state[index], 1.4, 0.05)) + ONE_HUNDRED


class HumanoidHeightNormalNarrowLowPenaltyShiftedOneHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (50 * normal_dist_density(state[index], 1.4, 0.01)) + ONE_HUNDRED


class HumanoidHeightNormalWideLowPenaltyShiftedOneHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (5 * normal_dist_density(state[index], 1.4, 0.1)) + ONE_HUNDRED


############################ STANDARD ############################

class HumanoidHeightNormalStandardPenaltyShiftedOneHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (30 * normal_dist_density(state[index], 1.4, 0.05)) + ONE_HUNDRED


class HumanoidHeightNormalNarrowStandardPenaltyShiftedOneHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (150 * normal_dist_density(state[index], 1.4, 0.01)) + ONE_HUNDRED


class HumanoidHeightNormalWideStandardPenaltyShiftedOneHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (15 * normal_dist_density(state[index], 1.4, 0.1)) + ONE_HUNDRED


############################ HIGH ############################


class HumanoidHeightNormalHighPenaltyShiftedOneHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (60 * normal_dist_density(state[index], 1.4, 0.05)) + ONE_HUNDRED


class HumanoidHeightNormalNarrowHighPenaltyShiftedOneHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (300 * normal_dist_density(state[index], 1.4, 0.01)) + ONE_HUNDRED


class HumanoidHeightNormalWideHighPenaltyShiftedOneHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (30 * normal_dist_density(state[index], 1.4, 0.1)) + ONE_HUNDRED

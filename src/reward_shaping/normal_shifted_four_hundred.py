from src.constants import Constants
from src.reward_shaping.fi_base import Fi
from src.reward_shaping.util import normal_dist_density

FOUR_HUNDRED = 400


############################ SMALL ############################
class HumanoidHeightNormalSmallPenaltyShiftedFourHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (5 * normal_dist_density(state[index], 1.4, 0.05)) + FOUR_HUNDRED


class HumanoidHeightNormalNarrowSmallPenaltyShiftedFourHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (25 * normal_dist_density(state[index], 1.4, 0.01)) + FOUR_HUNDRED


class HumanoidHeightNormalWideSmallPenaltyShiftedFourHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (2.5 * normal_dist_density(state[index], 1.4, 0.1)) + FOUR_HUNDRED


############################ LOW  ############################
class HumanoidHeightNormalLowPenaltyShiftedFourHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (10 * normal_dist_density(state[index], 1.4, 0.05)) + FOUR_HUNDRED


class HumanoidHeightNormalNarrowLowPenaltyShiftedFourHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (50 * normal_dist_density(state[index], 1.4, 0.01)) + FOUR_HUNDRED


class HumanoidHeightNormalWideLowPenaltyShiftedFourHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (5 * normal_dist_density(state[index], 1.4, 0.1)) + FOUR_HUNDRED


############################ STANDARD ############################

class HumanoidHeightNormalStandardPenaltyShiftedFourHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (30 * normal_dist_density(state[index], 1.4, 0.05)) + FOUR_HUNDRED


class HumanoidHeightNormalNarrowStandardPenaltyShiftedFourHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (150 * normal_dist_density(state[index], 1.4, 0.01)) + FOUR_HUNDRED


class HumanoidHeightNormalWideStandardPenaltyShiftedFourHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (15 * normal_dist_density(state[index], 1.4, 0.1)) + FOUR_HUNDRED


############################ HIGH ############################


class HumanoidHeightNormalHighPenaltyShiftedFourHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (60 * normal_dist_density(state[index], 1.4, 0.05)) + FOUR_HUNDRED


class HumanoidHeightNormalNarrowHighPenaltyShiftedFourHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (300 * normal_dist_density(state[index], 1.4, 0.01)) + FOUR_HUNDRED


class HumanoidHeightNormalWideHighPenaltyShiftedFourHundred(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (30 * normal_dist_density(state[index], 1.4, 0.1)) + FOUR_HUNDRED

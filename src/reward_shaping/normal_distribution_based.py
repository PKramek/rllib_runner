# Current Best
import numpy as np

from src.constants import Constants
from src.reward_shaping.fi_base import Fi
from src.reward_shaping.util import normal_dist_density


class HumanoidHeightNormalLowPenaltyShiftedExtremely(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return (10 * normal_dist_density(state[index], 1.4, 0.05)) + 500


class HumanoidFromSeminaryBest(Fi):
    def __call__(self, state: np.ndarray):
        return self._base_penalty(state) + 500

    def _base_penalty(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        return 10 * normal_dist_density(state[index], middle_of_dist, 0.05)


class HumanoidFromSeminaryBestBiggerDifferences(HumanoidFromSeminaryBest):
    def _base_penalty(self, state: np.ndarray):
        return super()._base_penalty(state) * 1.5


class HumanoidFromSeminaryBestEvenBiggerDifferences(HumanoidFromSeminaryBest):
    def _base_penalty(self, state: np.ndarray):
        return super()._base_penalty(state) * 2.0


class HumanoidFromSeminaryBestSmallerDifferences(HumanoidFromSeminaryBest):
    def _base_penalty(self, state: np.ndarray):
        return super()._base_penalty(state) * 0.7


class HumanoidFromSeminaryBestEvenSmallerDifferences(HumanoidFromSeminaryBest):
    def _base_penalty(self, state: np.ndarray):
        return super()._base_penalty(state) * 0.5

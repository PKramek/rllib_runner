import numpy as np
from scipy.stats import t

from src.constants import Constants
from src.reward_shaping.fi_base import Fi


class TStudentHeightNormal(Fi):
    def __call__(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        degree_of_freedom = 0.5
        scale = 0.05

        return t.pdf(state[index], df=degree_of_freedom, scale=scale, loc=middle_of_dist)


class TStudentHeightLowPenaltyShiftedFiveHundred(Fi):
    def __call__(self, state: np.ndarray):
        return self._base_penalty(state) + 500

    def _base_penalty(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        degree_of_freedom = 0.01
        scale = 0.35

        return 10 * t.pdf(state[index], df=degree_of_freedom, scale=scale, loc=middle_of_dist)


class TStudentHeightLowPenaltyBiggerDifferenceShiftedFiveHundred(TStudentHeightLowPenaltyShiftedFiveHundred):
    def _base_penalty(self, state: np.ndarray):
        return super()._base_penalty(state) * 1.5


class TStudentHeightLowPenaltyEvenBiggerDifferenceShiftedFiveHundred(TStudentHeightLowPenaltyShiftedFiveHundred):
    def _base_penalty(self, state: np.ndarray):
        return super()._base_penalty(state) * 2.0


class TStudentHeightLowPenaltySmallerDifferenceShiftedFiveHundred(TStudentHeightLowPenaltyShiftedFiveHundred):
    def _base_penalty(self, state: np.ndarray):
        return super()._base_penalty(state) * 0.7


class TStudentHeightLowPenaltyEvenSmallerDifferenceShiftedFiveHundred(TStudentHeightLowPenaltyShiftedFiveHundred):
    def _base_penalty(self, state: np.ndarray):
        return super()._base_penalty(state) * 0.5

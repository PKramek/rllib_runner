from typing import Type

import numpy as np

from src.reward_shaping.fi_base import Fi, AbstractHumanoidMultipleDimensions
from src.reward_shaping.normal_distribution_based import HumanoidHeightNormalLowPenaltyShiftedExtremely, \
    HumanoidFromSeminaryBestBiggerDifferences, HumanoidFromSeminaryBestEvenBiggerDifferences, \
    HumanoidFromSeminaryBestSmallerDifferences, HumanoidFromSeminaryBestEvenSmallerDifferences, HumanoidFromSeminaryBest
# This class is just an example and is not useful in any way
from src.reward_shaping.t_student_based import TStudentHeightLowPenaltyShiftedFiveHundred


class SumFi(Fi):
    def __call__(self, state):
        return np.sum(state)


class HumanoidJustFive(AbstractHumanoidMultipleDimensions):
    def _height_penalty(self, state):
        return 500.0

    def _forward_tilt_penalty(self, state):
        return 0.0

    def _x_axis_angle_rotation_penalty(self, state):
        return 0.0


class FiFactory:
    FI_MAPPING = {
        "normalSmallSHiftedExtremally": HumanoidHeightNormalLowPenaltyShiftedExtremely,
        "justFiveHundred": HumanoidJustFive,

        "fromSeminary": HumanoidFromSeminaryBest,
        "fromSeminaryBiggerDifferences": HumanoidFromSeminaryBestBiggerDifferences,
        "fromSeminaryEvenBiggerDifferences": HumanoidFromSeminaryBestEvenBiggerDifferences,
        "fromSeminarySmallerDifferences": HumanoidFromSeminaryBestSmallerDifferences,
        "fromSeminaryEvenSmallerDifferences": HumanoidFromSeminaryBestEvenSmallerDifferences,

        "tStudent": TStudentHeightLowPenaltyShiftedFiveHundred,
        "tStudentBiggerDifferences": HumanoidFromSeminaryBestBiggerDifferences,
        "tStudentEvenBiggerDifferences": HumanoidFromSeminaryBestEvenBiggerDifferences,
        "tStudentSmallerDifferences": HumanoidFromSeminaryBestSmallerDifferences,
        "tStudentEvenSmallerDifferences": HumanoidFromSeminaryBestEvenSmallerDifferences,
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

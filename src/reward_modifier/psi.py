from typing import Type

import numpy as np

from src.constants import Constants
from src.reward_modifier.psi_base import Psi
from src.reward_modifier.util import normal_dist_density


class AliveBonus(Psi):
    def __call__(self, state: np.ndarray) -> float:
        return 5.0


class AlivePenalty(Psi):
    def __call__(self, state: np.ndarray) -> float:
        return -5.0


class HeightBonusNarrow(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 80 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.02)


class SmallerHeightBonusNarrow(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 60 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.02)


class SmallHeightBonusNarrow(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 40 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.02)


class HeightBonus(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 32 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.05)


class SmallerHeightBonus(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 20 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.05)


class SmallHeightBonus(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 16 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.05)


class HeightPenaltyNarrow(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 80 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.02) - 5


class SmallerHeightPenaltyNarrow(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 60 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.02) - 3.8


class SmallHeightPenaltyNarrow(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 40 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.02) - 2.51


class HeightPenalty(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 32 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.05) - 5.0


class SmallerHeightPenalty(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 20 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.05) - 3.14


class SmallHeightPenalty(Psi):
    def __call__(self, state: np.ndarray) -> float:
        index = Constants.HEIGHT_INDEX
        return 16 * normal_dist_density(state[index], Constants.HEIGHT_NOMINAL_VALUE, 0.05) - 2.51


class PsiFactory:
    PSI_MAPPING = {
        'aliveBonus': AliveBonus,
        "alivePenalty": AlivePenalty,

        "heightBonus": HeightBonus,
        "smallerHeightBonus": SmallerHeightBonus,
        "smallHeightBonus": SmallHeightBonus,

        "heightNarrowBonus": HeightBonusNarrow,
        "smallerHeightNarrowBonus": SmallerHeightBonusNarrow,
        "smallHeightNarrowBonus": SmallHeightBonusNarrow,

        # Penalty
        "heightPenalty": HeightPenalty,
        "smallerHeightPenalty": SmallerHeightPenalty,
        "smallHeightPenalty": SmallHeightPenalty,

        "heightNarrowPenalty": HeightPenaltyNarrow,
        "smallerHeightNarrowPenalty": SmallerHeightPenaltyNarrow,
        "smallHeightNarrowPenalty": SmallHeightPenaltyNarrow,
    }

    @staticmethod
    def get_psi(name: str):
        fi = PsiFactory.PSI_MAPPING.get(name, None)

        if fi is None:
            raise ValueError(f"Unknown fi: {name}, viable options are: {PsiFactory.PSI_MAPPING.keys()}")

        return fi()

    @staticmethod
    def register(name: str, _class=Type[Psi]):
        assert issubclass(_class, Psi), "Can only register classes that are subclasses of Fi"
        assert name not in PsiFactory.PSI_MAPPING, "This name is already taken"

        PsiFactory.PSI_MAPPING[name] = _class

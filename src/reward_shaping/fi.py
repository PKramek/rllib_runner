from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from src.constants import Constants


class Fi(ABC):
    @abstractmethod
    def __call__(self, state):
        pass


# This class is just an example and is not useful in any way
class SumFi(Fi):
    def __call__(self, state):
        return np.sum(state)


class HumanoidHeightLinear(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.35 < state[index] < 1.45 else -np.abs((1.4 - state[index]) * 10)


class HumanoidHeightLinearLowerPenalty(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.35 < state[index] < 1.45 else -np.abs((1.4 - state[index]) * 5)


class HumanoidHeightLinearHigherPenalty(Fi):
    def __call__(self, state):
        index = Constants.HEIGHT_INDEX
        return 0.0 if 1.35 < state[index] < 1.45 else -np.abs((1.4 - state[index]) * 50)


class HumanoidHeightLinearShiftedDown(HumanoidHeightLinear):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi - 1


class HumanoidHeightLinearLowerPenaltyShiftedDown(HumanoidHeightLinearLowerPenalty):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi - 1


class HumanoidHeightLinearHigherPenaltyShiftedDown(HumanoidHeightLinearHigherPenalty):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi - 1


class HumanoidHeightLinearShiftedUp(HumanoidHeightLinear):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi + 5


class HumanoidHeightLinearShiftedUpSlightly(HumanoidHeightLinear):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi + 1


class HumanoidHeightLinearShiftedUpMassively(HumanoidHeightLinear):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi + 20


class HumanoidHeightLinearHigherPenaltyShiftedUp(HumanoidHeightLinearHigherPenalty):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi + 5


class HumanoidHeightLinearHigherPenaltyShiftedUpSlightly(HumanoidHeightLinearHigherPenalty):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi + 1


class HumanoidHeightLinearHigherPenaltyShiftedUpMassively(HumanoidHeightLinearHigherPenalty):
    def __call__(self, state):
        base_fi = super().__call__(state)
        return base_fi + 20


class FiFactory:
    FI_MAPPING = {
        'linear': HumanoidHeightLinear,
        'linearLowerPenalty': HumanoidHeightLinearLowerPenalty,
        'linearHigherPenalty': HumanoidHeightLinearHigherPenalty,
        'linearShiftedDown': HumanoidHeightLinearShiftedDown,
        'linearLowerPenaltyShiftedDown': HumanoidHeightLinearLowerPenaltyShiftedDown,
        'linearHigherPenaltyShiftedDown': HumanoidHeightLinearHigherPenaltyShiftedDown,
        'linearShiftedUp': HumanoidHeightLinearShiftedUp,
        'linearShiftedUpSlightly': HumanoidHeightLinearShiftedUpSlightly,
        'linearShiftedUpMassively': HumanoidHeightLinearShiftedUpMassively,
        'linearHigherPenaltyShiftedUp': HumanoidHeightLinearHigherPenaltyShiftedUp,
        'linearHigherPenaltyShiftedUpSlightly': HumanoidHeightLinearHigherPenaltyShiftedUpSlightly,
        'linearHigherPenaltyShiftedUpMassively': HumanoidHeightLinearHigherPenaltyShiftedUpMassively
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

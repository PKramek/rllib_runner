from typing import Type

import numpy as np

from src.reward_modifier.psi_base import Psi


class AliveBonus(Psi):
    def __call__(self, state: np.ndarray) -> float:
        return 5.0


class PsiFactory:
    PSI_MAPPING = {
        'aliveBonus': AliveBonus
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

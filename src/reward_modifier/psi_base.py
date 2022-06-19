from abc import ABC, abstractmethod


class Psi(ABC):
    @abstractmethod
    def __call__(self, state) -> float:
        pass

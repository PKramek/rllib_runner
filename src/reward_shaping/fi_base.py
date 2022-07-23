from abc import ABC, abstractmethod


class Fi(ABC):
    @abstractmethod
    def __call__(self, state):
        pass


class AbstractHumanoidMultipleDimensions(ABC):
    def __call__(self, state):
        return self._height_penalty(state) + self._forward_tilt_penalty(state) + self._x_axis_angle_rotation_penalty(
            state)

    @abstractmethod
    def _height_penalty(self, state):
        raise NotImplementedError

    @abstractmethod
    def _forward_tilt_penalty(self, state):
        raise NotImplementedError

    @abstractmethod
    def _x_axis_angle_rotation_penalty(self, state):
        raise NotImplementedError

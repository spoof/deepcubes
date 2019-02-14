from typing import List

from ..cubes import Cube


class EmbedderFactoryABC(object):
    def create(self, mode):
        raise NotImplementedError


class Embedder(Cube):

    def __init__(self, mode):
        self.mode = mode

    def save(self):
        cube_params = {
            'class': self.__class__.__name__,
            'mode': self.mode
        }

        return cube_params

    def forward(self, tokens) -> List[float]:
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

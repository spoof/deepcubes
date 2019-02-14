from ..cubes import Cube

from typing import List


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

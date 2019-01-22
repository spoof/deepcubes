from collections import namedtuple
from typing import List
import os


class Cube(object):
    """Base class for all algorithmical blocks in framework"""

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(*input)

    def save(self, path, name, *input):
        os.makedirs(path, exist_ok=True)

    def load(self, *input):
        raise NotImplementedError


class TrainableCube(Cube):
    """Allow train cube with some data"""

    def train(self, *input):
        raise NotImplementedError


CubeLabel = namedtuple("CubeLabel", "label, proba")


class PredictorCube(Cube):
    """Cube that return labels and probas"""

    def forward(self, *input) -> List[CubeLabel]:
        raise NotImplementedError

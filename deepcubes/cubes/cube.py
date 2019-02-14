from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List


class Cube(ABC):
    """Base class for all algorithmical blocks in framework"""

    @abstractmethod
    def forward(self, *input):
        ...

    def __call__(self, *input):
        return self.forward(*input)


class Serializable(ABC):
    """Absractr class for object that can be serialized and deserialized back"""

    # TODO: rename methods to serialize/deserialize
    @abstractmethod
    def save(self, path, name, *input):
        ...

    @staticmethod
    @abstractmethod
    def load(cls, *input):
        ...


class Trainable(ABC):
    """Allow train cube with some data"""

    @abstractmethod
    def train(self, *input):
        ...


CubeLabel = namedtuple("CubeLabel", "label, proba")


class Predictor(ABC):
    """Cube that return labels and probas"""

    @abstractmethod
    def forward(self, *input) -> List[CubeLabel]:
        raise NotImplementedError

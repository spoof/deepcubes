from .cube import TrainableCube
from gensim.models import KeyedVectors
import numpy as np
import json
import os


class Embedder(TrainableCube):
    """Word embedder"""

    def __init__(self):
        self.path = None
        self.model = None

    def train(self, path):
        self.path = path
        self.model = KeyedVectors.load(path, mmap='r')

    def forward(self, tokens):
        """Calculate vector for sentence as mean vector of all its words"""

        vector = np.zeros(self.model.vector_size)
        words_in_model = 0

        for word in tokens:
            if word in self.model:
                vector += self.model.get_vector(word)
                words_in_model += 1

        if words_in_model:
            vector /= words_in_model

        return vector

    def save(self, path, name='embedder.cube'):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'path': self.path
        }

        cube_path = os.path.join(path, name)
        with open(cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return cube_path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        embedder = cls()

        path = cube_params['path']
        if path:
            embedder.train(path)

        return embedder

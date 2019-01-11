from .cube import Cube
from gensim.models import KeyedVectors
import numpy as np
import json
import os


class Embedder(Cube):
    """Word embedder"""

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

    def train(self, path):
        self.path = path
        self.model = KeyedVectors.load(path, mmap='r')

    def save(self, name='embedder.cube', path='scripts/embedders'):
        os.makedirs(path, exist_ok=True)
        cube_params = {
            'cube': self.__class__.__name__,
            'path': self.path
        }
        with open(os.path.join(path, name), 'w') as out:
            out.write(json.dumps(cube_params))

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())
        path = cube_params['path']
        embedder = cls()
        embedder.train(path)
        return embedder

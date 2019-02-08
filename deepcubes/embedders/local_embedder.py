from gensim.models import KeyedVectors
import numpy as np
import json
import os

from ..embedders import Embedder


class LocalEmbedder(Embedder):
    """Word embedder"""

    def __init__(self, matrix_path, mode=None):
        if mode is None:
            mode = os.path.splitext(os.path.basename(matrix_path))[0]

        super().__init__(mode)
        self.matrix_path = matrix_path
        self.model = KeyedVectors.load(matrix_path, mmap='r')

    def __call__(self, *input):
        return self.forward(*input)

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

    @classmethod
    def load(cls, path, matrix_path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        network_embedder = cls(matrix_path, cube_params["mode"])
        return network_embedder

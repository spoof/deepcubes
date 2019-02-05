from gensim.models import KeyedVectors
import numpy as np


class Embedder(object):
    """Word embedder"""

    def __init__(self, matrix_path):
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

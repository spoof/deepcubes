from .cube import TrainableCube

import editdistance as ed


class Corrector(TrainableCube):
    """Fix text to nearest vocabulary values"""

    def __init__(self):
        self.vocab = []
        self.max_distance = -1

    def forward(self, tokens):
        corrected_tokens = []

        for token in tokens:

            nearest_word, min_dist = None, None
            for word in self.vocab:
                dist = ed.eval(word, token)

                if dist > self.max_distance:
                    continue

                if min_dist is None or dist < min_dist:
                    nearest_word, min_dist = word, dist

            if nearest_word is not None:
                corrected_tokens.append(nearest_word)
            else:
                corrected_tokens.append(token)

        return corrected_tokens

    def train(self, vocab, max_distance):
        self.vocab = vocab
        self.max_distance = max_distance

    def save(self):
        cube_params = {
            'class': self.__class__.__name__,
            'vocab': self.vocab,
            'max_distance': self.max_distance,
        }

        return cube_params

    @classmethod
    def load(cls, cube_params):
        corrector = cls()
        corrector.train(cube_params["vocab"],
                        cube_params["max_distance"])

        return corrector

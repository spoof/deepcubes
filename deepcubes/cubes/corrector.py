from .cube import TrainableCube

import editdistance as ed
import json
import os


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

    def save(self, path, name="corrector.cube"):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'vocab': self.vocab,
            'max_distance': self.max_distance,
        }

        cube_path = os.path.join(path, name)

        with open(cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return cube_path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        corrector = cls()
        corrector.train(cube_params["vocab"],
                        cube_params["max_distance"])

        return corrector

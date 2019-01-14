from deepcubes.cubes import Cube
import re


class PatternMatcher(Cube):
    """Matcher based on regexps"""

    def __init__(self):
        self.data = []

    def train(self, labels, labels_patterns):
        self.data = []
        for label, patterns in zip(labels, labels_patterns):
            self.data.append((label, patterns))

    def predict(self, query):
        labels, probas = [], []
        for label, patterns in self.data:

            proba = 0
            for pattern in patterns:
                if re.match(pattern, query):
                    proba = 1
                    break

            labels.append(label)
            probas.append(proba)

        return labels, probas

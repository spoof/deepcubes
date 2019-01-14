from deepcubes.cubes import TrainableCube, PredictorCube, CubeLabel
import re


class PatternMatcher(TrainableCube, PredictorCube):
    """Matcher based on regexps"""

    def __init__(self):
        self.data = []

    def train(self, labels, labels_patterns):
        self.data = []
        for label, patterns in zip(labels, labels_patterns):
            self.data.append((label, patterns))

    def forward(self, query):
        labels, probas = [], []
        for label, patterns in self.data:

            proba = 0
            for pattern in patterns:
                if re.match(pattern, query):
                    proba = 1
                    break

            labels.append(label)
            probas.append(proba)

        return [CubeLabel(label, proba)
                for label, proba in zip(labels, probas)]

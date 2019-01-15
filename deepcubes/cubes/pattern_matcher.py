from deepcubes.cubes import TrainableCube, PredictorCube, CubeLabel
import re
from collections import defaultdict


class PatternMatcher(TrainableCube, PredictorCube):
    """Matcher based on regexps"""

    def __init__(self):
        self.data = []

    def train(self, labels, patterns):
        """Arguments:

            labels:  [[..], [..]]  nested lists of patterns
            patterns: [[..], [..]]  nested list of corresponded labels
        """

        self.labels = labels
        self.patterns = patterns

    def forward(self, query):
        unique_labels = set()
        labels_probas = defaultdict(int)

        for labels, patterns in zip(self.labels, self.patterns):
            unique_labels.update(labels)
            for pattern in patterns:
                if re.match(pattern, query):
                    for label in labels:
                        labels_probas[label] = 1

                    break

        return sorted([CubeLabel(label, labels_probas[label])
                       for label in unique_labels],
                      key=lambda elem: (-elem[1], elem[0]))

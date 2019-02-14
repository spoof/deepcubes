from .cube import TrainableCube, PredictorCube, CubeLabel
from ..utils.functions import sorted_labels

import re
from collections import defaultdict


class PatternMatcher(TrainableCube, PredictorCube):
    """Matcher based on regexps"""

    def __init__(self):
        self.labels = []
        self.patterns = []

    def train(self, labels, patterns):
        """Arguments:

            labels:  [[..], [..]]  nested lists of patterns
            patterns: [[..], [..]]  nested list of corresponded labels
        """

        self.labels = labels
        self.patterns = [[p.lower() for p in ptrns] for ptrns in patterns]

    def forward(self, query):
        unique_labels = set()
        labels_probas = defaultdict(int)
        prepared_query = query.strip().lower()

        for labels, patterns in zip(self.labels, self.patterns):
            unique_labels.update(labels)
            for pattern in patterns:
                if re.match(pattern, prepared_query):
                    for label in labels:
                        labels_probas[label] = 1

                    break

        return sorted_labels([CubeLabel(label, labels_probas[label])
                              for label in unique_labels])

    def save(self):
        cube_params = {
            'class': self.__class__.__name__,
            'labels': self.labels,
            'patterns': self.patterns,
        }

        return cube_params

    @classmethod
    def load(cls, cube_params):
        pattern_matcher = cls()
        pattern_matcher.train(cube_params["labels"], cube_params["patterns"])

        return pattern_matcher

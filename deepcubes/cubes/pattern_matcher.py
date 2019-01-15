from .cube import TrainableCube, PredictorCube, CubeLabel
from ..utils.functions import sorted_labels

import re
import os
import json
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

    def save(self, path, name="pattern_matcher.cube"):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'labels': self.labels,
            'patterns': self.patterns,
        }

        cube_path = os.path.join(path, name)

        with open(cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return cube_path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        pattern_matcher = cls()
        pattern_matcher.train(cube_params["labels"], cube_params["patterns"])

        return pattern_matcher

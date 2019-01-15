from .cube import TrainableCube, PredictorCube, CubeLabel
from ..utils.functions import sorted_labels

import editdistance as ed
import os
import json
from collections import defaultdict


class EditDistanceMatcher(PredictorCube, TrainableCube):
    """Matcher based on edit distance"""

    def __init__(self):
        self.data = []
        self.max_distance = -1
        self.labels = []
        self.texts = []

    def train(self, labels, texts, max_distance):
        """Arguments:

            texts:  [[..], [..]]  nested lists of texts
            labels: [[..], [..]]  nested list of corresponded labels
            max_distance: maximal edit distance to assign label
        """

        self.labels = labels
        self.texts = texts
        self.max_distance = max_distance

    def forward(self, query):
        unique_labels = set()
        labels_probas = defaultdict(int)

        for labels, texts in zip(self.labels, self.texts):
            unique_labels.update(labels)
            for text in texts:
                dist = ed.eval(query, text)

                if dist <= self.max_distance:
                    for label in labels:
                        labels_probas[label] = 1

                    break

        return sorted_labels([CubeLabel(label, labels_probas[label])
                              for label in unique_labels])

    def save(self, path, name="edit_distance_matcher.cube"):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'labels': self.labels,
            'texts': self.texts,
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

        edit_distance_matcher = cls()
        edit_distance_matcher.train(cube_params["labels"],
                                    cube_params["texts"],
                                    cube_params["max_distance"])

        return edit_distance_matcher

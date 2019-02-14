from .cube import TrainableCube, PredictorCube, CubeLabel
from ..utils.functions import sorted_labels

import editdistance as ed
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
        self.texts = [[text.lower() for text in txt] for txt in texts]
        self.max_distance = max_distance

    def forward(self, query):
        unique_labels = set()
        labels_probas = defaultdict(int)
        prepared_query = query.lower()

        for labels, texts in zip(self.labels, self.texts):
            unique_labels.update(labels)
            for text in texts:
                dist = ed.eval(prepared_query, text)

                if dist <= self.max_distance:
                    for label in labels:
                        labels_probas[label] = 1

                    break

        return sorted_labels([CubeLabel(label, labels_probas[label])
                              for label in unique_labels])

    def save(self):
        cube_params = {
            'class': self.__class__.__name__,
            'labels': self.labels,
            'texts': self.texts,
            'max_distance': self.max_distance,
        }

        return cube_params

    @classmethod
    def load(cls, cube_params):
        edit_distance_matcher = cls()
        edit_distance_matcher.train(cube_params["labels"],
                                    cube_params["texts"],
                                    cube_params["max_distance"])

        return edit_distance_matcher

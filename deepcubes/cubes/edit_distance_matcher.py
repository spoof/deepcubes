from deepcubes.cubes import Cube
import editdistance as ed
from collections import namedtuple


EditDistanceLabel = namedtuple("EditDistanceLabel", ["label", "texts"])


class EditDistanceMatcher(Cube):
    """Matcher based on edit distance"""

    def __init__(self):
        self.data = []
        self.max_distance = -1

    def train(self, labels, labels_texts, max_distance):
        for label, texts in zip(labels, labels_texts):
            self.data.append(EditDistanceLabel(label, texts))

        self.max_distance = max_distance

    def predict(self, query):
        labels, probas = [], []

        for label, texts in self.data:

            nearest_dist = None
            for text in texts:
                dist = ed.eval(query, text)

                if dist > self.max_distance:
                    continue

                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist

            labels.append(label)
            probas.append(int(nearest_dist is not None))

        return labels, probas

from deepcubes.cubes import TrainableCube, PredictorCube, CubeLabel
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

        return sorted([CubeLabel(label, labels_probas[label])
                       for label in unique_labels],
                      key=lambda elem: (-elem[1], elem[0]))

    def save(self, path, name="pattern_matcher"):
        return None

    def load(self):
        return None

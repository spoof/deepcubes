from deepcubes.cubes import Cube
import editdistance as ed


class EditDistanceMatcher(Cube):
    """Matcher based on edit distance"""

    def __init__(self):
        self.data = []
        self.max_distance = -1

    def train(self, labels, labels_texts, max_distance):
        self.data = []
        for label, texts in zip(labels, labels_texts):
            self.data.append((label, texts))

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

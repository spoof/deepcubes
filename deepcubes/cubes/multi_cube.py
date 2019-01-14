from . import Cube


class MultiCube(Cube):
    """Cube that apply different cubes for different labels"""

    def __init__(self):
        self.data = []

    def train(self, labels, labels_cubes):
        self.data = []
        for label, cubes in zip(labels, labels_cubes):
            self.data.append(label, cubes)

    def predict(self, query):
        pass

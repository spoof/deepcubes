import json
import os

from ..cubes import TrainableCube, PredictorCube
from ..cubes import EditDistanceMatcher


class Sentiment(TrainableCube, PredictorCube):

    MAX_EDIT_DISTANCE = 1

    def __init__(self):
        self.classifier = EditDistanceMatcher()

    def forward(self, query):
        return self.classifier(query)

    def train(self, labels, texts):
        self.classifier.train(labels, texts, self.MAX_EDIT_DISTANCE)

    def save(self, path, name='intent_classifier.cube'):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'classifier': self.classifier.save(path=path),
        }

        self.cube_path = os.path.join(path, name)
        with open(self.cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return self.cube_path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        model = cls()
        model.classifier = EditDistanceMatcher.load(cube_params["classifier"])

        return model

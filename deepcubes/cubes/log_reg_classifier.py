from .cube import CubeLabel, PredictorCube, TrainableCube
from ..utils.functions import logistic_regression_from_dict
from ..utils.functions import logistic_regression_to_dict

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

import numpy as np
import json
import os


class LogRegClassifier(PredictorCube, TrainableCube):
    """Classify"""

    def __init__(self, solver='liblinear', multi_class='ovr'):
        self.clf = LogisticRegression(solver=solver, multi_class=multi_class)

    def train(self, X, Y):
        """Train classifier at question-answer pairs"""
        self.clf.fit(X, Y)

    def forward(self, vector):
        try:
            probas = self.clf.predict_proba([vector])[0]
            order = np.argsort(probas)[::-1]

            return [
                CubeLabel(self.clf.classes_[label], probas[label])
                for label in order
            ]

        except NotFittedError as e:
            # TODO(dima): implement logic
            raise e

    def save(self, path, name='logistic_regression.cube'):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'clf': logistic_regression_to_dict(self.clf),
        }

        cube_path = os.path.join(path, name)
        with open(cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return cube_path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        classifier = cls()
        classifier.clf = logistic_regression_from_dict(cube_params['clf'])

        return classifier

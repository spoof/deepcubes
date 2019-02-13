from .cube import CubeLabel, PredictorCube, TrainableCube
from ..utils.functions import logistic_regression_from_dict
from ..utils.functions import logistic_regression_to_dict

from sklearn.linear_model import LogisticRegression

import numpy as np
import json
import os


class LogRegClassifier(PredictorCube, TrainableCube):
    """Classify"""

    def __init__(self, solver='liblinear', multi_class='ovr'):
        self.clf = LogisticRegression(solver=solver, multi_class=multi_class)
        self.trained = False
        self.single_label = None

    def train(self, X, Y):
        """Train classifier at question-answer pairs"""
        number_of_uniq_y = len(set(Y))
        self.single_label = None

        # not fit in case of no data
        if not number_of_uniq_y:
            self.trained = False
            return

        # in case of just one label
        if number_of_uniq_y == 1:
            self.single_label = Y[0]
            self.trained = True
            return

        # normal logistic regression
        self.clf.fit(X, Y)
        self.trained = True

    def forward(self, vector):
        if not self.trained:
            return []

        if self.single_label:
            return [CubeLabel(self.single_label, 1)]

        probas = self.clf.predict_proba([vector])[0]
        order = np.argsort(probas)[::-1]

        return [
            CubeLabel(self.clf.classes_[label], probas[label])
            for label in order
        ]

    def save(self, path, name='logistic_regression.cube'):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'trained': self.trained,
            'single_label': self.single_label,
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
        classifier.trained = cube_params["trained"]
        classifier.single_label = cube_params["single_label"]

        return classifier

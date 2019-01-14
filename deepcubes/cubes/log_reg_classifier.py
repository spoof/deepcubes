from .cube import CubeLabel, PredictorCube, TrainableCube
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

import numpy as np
import pickle
import os


class LogRegClassifier(PredictorCube, TrainableCube):
    """Classify"""

    def __init__(self, solver='liblinear', multi_class='ovr'):
        self.clf = LogisticRegression(
            solver=solver,
            multi_class=multi_class,
        )

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

    def save(
        self, name='logistic_regression.cube', path='scripts/classifiers'
    ):
        os.makedirs(path, exist_ok=True)
        cube_path = os.path.join(path, name)
        with open(cube_path, "wb") as handle:
            pickle.dump(
                {
                    'cube': self.__class__.__name__,
                    'clf': self.clf,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
                file=handle
            )
        return cube_path

    @classmethod
    def load(cls, path):
        with open(path, "rb") as handle:
            data = pickle.load(handle)

        classifier = cls()
        classifier.clf = data['clf']

        return classifier

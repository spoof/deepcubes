import numpy as np
from sklearn.linear_model import LogisticRegression

from ..utils.functions import logistic_regression_from_dict, logistic_regression_to_dict
from .cube import Cube, CubeLabel, Predictor, Serializable, Trainable


class LogRegClassifier(Cube, Predictor, Trainable, Serializable):
    """Classify"""

    def __init__(self, solver='liblinear', multi_class='ovr'):
        self.clf = LogisticRegression(solver=solver, multi_class=multi_class)
        self.single_label = None
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

    def save(self):
        cube_params = {
            'class': self.__class__.__name__,
            'trained': self.trained,
            'single_label': self.single_label,
            'clf': logistic_regression_to_dict(self.clf),
        }

        return cube_params

    @classmethod
    def load(cls, cube_params):
        classifier = cls()
        classifier.clf = logistic_regression_from_dict(cube_params['clf'])
        classifier.trained = cube_params["trained"]
        classifier.single_label = cube_params["single_label"]

        return classifier

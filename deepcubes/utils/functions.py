import os
from typing import List
from ..cubes.cube import CubeLabel

import numpy as np
from sklearn.linear_model import LogisticRegression


def sorted_labels(cube_labels: List[CubeLabel]):
    return sorted(cube_labels, key=lambda elem: (-elem.proba, elem.label))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def logistic_regression_to_dict(lr_model):
    data = {
        'init_params': lr_model.get_params(),
        'model_params': {}
    }

    for p in ('coef_', 'intercept_', 'classes_', 'n_iter_'):
        data['model_params'][p] = getattr(lr_model, p).tolist()

    return data


def logistic_regression_from_dict(data):
    model = LogisticRegression(**data['init_params'])
    for name, p in data['model_params'].items():
        setattr(model, name, np.array(p))

    return model


def is_url(path):
    # TODO: need more sophisticated url checker
    return path.startswith("http")


def get_new_model_id(path):
    models_ids = [int(file_name) for file_name in os.listdir(path) if not (
        os.path.isfile(os.path.join(path, file_name))
    )]

    sorted_ids = sorted(models_ids)
    new_model_id = sorted_ids[-1] + 1 if len(sorted_ids) else 0

    return new_model_id

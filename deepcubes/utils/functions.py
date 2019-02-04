from typing import List
from ..cubes.cube import CubeLabel

import numpy as np


def sorted_labels(cube_labels: List[CubeLabel]):
    return sorted(cube_labels, key=lambda elem: (-elem.proba, elem.label))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

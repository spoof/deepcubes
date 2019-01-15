from typing import List
from ..cubes.cube import CubeLabel


def sorted_labels(cube_labels: List[CubeLabel]):
    return sorted(cube_labels, key=lambda elem: (-elem.proba, elem.label))

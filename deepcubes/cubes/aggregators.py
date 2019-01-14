from .cube import PredictorCube, CubeLabel, Cube
from collections import defaultdict


class Pipe(Cube):
    """Linear cubes pipeline"""

    def __init__(self, cubes):
        self.cubes = cubes

    def forward(self, *input):
        result = input
        for cube in self.cubes:
            result = cube(result)

        return result


class Max(PredictorCube):
    """Select maximal proba for each label from multiple cubes results"""

    def __init__(self, predictor_cubes):
        self.cubes = predictor_cubes

    def forward(self, *input):
        max_label_proba = defaultdict(float)

        for cube in self.cubes:
            for cube_label in cube(*input):
                if cube_label.proba >= max_label_proba[cube_label.label]:
                    max_label_proba[cube_label.label] = cube_label.proba

        return [CubeLabel(label, proba)
                for label, proba in max_label_proba.items()]

from .cube import PredictorCube


class MaxCube(PredictorCube):
    """Predictor"""

    def __init__(self, predictor_cubes):
        self.cubes = predictor_cubes

    def forward(self, *input):
        pass

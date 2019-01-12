class Cube(object):
    """Base class for all algorithmical blocks in framework"""

    def train(self, *input):
        pass

    def predict(self, *input):
        pass

    def save(self, *input):
        pass

    def load(self, *input):
        pass

    def __call__(self, *input):
        return self.predict(*input)


class PredictorCube(Cube):
    """Cube that return labels and probas"""
    pass


class MaxPredictorCube(PredictorCube):
    """Cube that apply serveral cubes for label and return aggregated proba"""
    pass

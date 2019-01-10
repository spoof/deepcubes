class Model(object):
    """Base class for large models"""

    def predict(self, *input):
        pass

    def __call__(self, *input):
        return self.predict(*input)

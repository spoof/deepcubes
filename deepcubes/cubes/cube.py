class Cube(object):
    """Base class for all algorithmical blocks in framework"""

    def forward(self, *input):
        pass

    def save(self, *input):
        pass

    def load(self, *input):
        pass

    def __call__(self, *input):
        return self.forward(*input)

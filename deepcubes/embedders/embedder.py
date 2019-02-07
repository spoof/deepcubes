from ..cubes import Cube

import os
import json


class Embedder(Cube):

    def __init__(self, mode):
        self.mode = mode

    def save(self, path, name="embedder.cube"):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'mode': self.mode
        }

        cube_path = os.path.join(path, name)

        with open(cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return cube_path

    def load(self):
        raise NotImplementedError

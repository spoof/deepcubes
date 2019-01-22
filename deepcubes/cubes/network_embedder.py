from .cube import Cube

import requests
import json
import os


class NetworkEmbedder(Cube):
    """Network embedder"""

    def __init__(self, url, mode):
        self.url = url
        self.mode = mode

    def forward(self, tokens):
        params = {
            'tokens': tokens,
            'mode': self.mode,
        }

        response = requests.get(self.url, params)
        if response.status_code != 200:
            return None

        content = json.loads(response.text)
        if 'vector' not in content:
            # TODO: think about
            return None
        else:
            return content['vector']

    def save(self, path, name='network_embedder.cube'):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'url': self.url,
            'mode': self.mode,
        }

        cube_path = os.path.join(path, name)
        with open(cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return cube_path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        network_embedder = cls(cube_params["url"],
                               cube_params["mode"])

        return network_embedder

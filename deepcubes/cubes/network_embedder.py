from .cube import TrainableCube

import requests
import json
import os


class NetworkEmbedder(TrainableCube):
    """Network embedder"""

    def __init__(self, url):
        self.emb_url = url
        self.mode = None

    def train(self, mode):
        self.mode = mode

    def forward(self, tokens):
        params = {
            'tokens': tokens,
            'mode': self.mode,
        }

        response = requests.get(self.emb_url, params)
        if response.status_code != 200:
            return None

        content = json.loads(response.text)
        if 'vector' not in content:
            return None
        else:
            return content['vector']

    def save(self, path, name='network_embedder.cube'):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'url': self.emb_url,
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

        url = cube_params['url']
        mode = cube_params['mode']

        network_embedder = cls(url)
        network_embedder.train(mode)

        return network_embedder

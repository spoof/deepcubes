from .cube import TrainableCube
import requests
import json
import os


class NetworkEmbedder(TrainableCube):
    """Network embedder"""

    def __init__(self, url):
        self.url = url

    def train(self, tag):
        self.tag = tag

    def forward(self, tokens):
        params = {
            'tokens': tokens,
            'tag': self.tag,
        }

        response = requests.get(self.url, params)
        if response.status_code != 200:
            return None

        content = json.loads(response.text)
        if 'vector' not in content:
            return None
        else:
            return content['vector']

    def save(self, name='network_embedder.cube', path='scripts/embedders'):
        os.makedirs(path, exist_ok=True)
        cube_params = {
            'cube': self.__class__.__name__,
            'url': self.url,
            'tag': self.tag,
        }

        with open(os.path.join(path, name), 'w') as out:
            out.write(json.dumps(cube_params))

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        url = cube_params['url']
        tag = cube_params['tag']

        network_embedder = cls()
        network_embedder.train(url, tag)

        return network_embedder

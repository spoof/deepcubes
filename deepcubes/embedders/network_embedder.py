import requests
import json
import os

from ..embedders import Embedder


class NetworkEmbedder(Embedder):
    """Network embedder"""

    EMPTY_STRING = ""

    def __init__(self, url, mode):
        if not mode:
            mode = os.path.basename(url)

        super().__init__(mode)

        self.url = url

    def __call__(self, *input):
        return self.forward(*input)

    def forward(self, tokens):
        # TODO: fix this not idiomatic way to process empty tokens
        if not len(tokens):
            tokens = [self.EMPTY_STRING]

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

    @classmethod
    def load(cls, path, url):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        network_embedder = cls(url, cube_params["mode"])
        return network_embedder

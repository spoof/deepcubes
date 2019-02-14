import json
import os
from enum import Enum

import requests

from deepcubes.embedders import (
    Embedder,
    EmbedderFactory as EmbedderFactoryABC,
    LocalEmbedder,
)


class FactoryType(Enum):
    LOCAL = 0
    NETWORK = 1


class EmbedderFactory(EmbedderFactoryABC):

    def __init__(self, path):
        if is_url(path):
            self.factory_type = FactoryType.NETWORK
        else:
            self.factory_type = FactoryType.LOCAL

        self.path = path

    def _get_full_url(self, mode):
        return "{}/{}".format(self.path, mode)

    def _get_full_path(self, mode):
        return os.path.join(self.path, "{}.kv".format(mode))

    def create(self, mode):
        if self.factory_type == FactoryType.NETWORK:
            return NetworkEmbedder(self._get_full_url(mode), mode)
        else:
            return LocalEmbedder(self._get_full_path(mode), mode)


class NetworkEmbedder(Embedder):
    """Network embedder"""

    EMPTY_STRING = ""

    def __init__(self, url, mode=None):
        if mode is None:
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
    def load(cls, cube_params, url):
        network_embedder = cls(url, cube_params["mode"])
        return network_embedder


def is_url(path):
    # TODO: need more sophisticated url checker
    return path.startswith("http")

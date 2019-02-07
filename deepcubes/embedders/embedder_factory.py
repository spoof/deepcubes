import os
import json
from enum import Enum

from ..embedders import LocalEmbedder, NetworkEmbedder


class FactoryType(Enum):
    LOCAL = 0
    NETWORK = 1


class EmbedderFactory(object):

    def __init__(self, path):
        # TODO: need more sophisticated url checker
        if "http" in path:
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
            return NetworkEmbedder(self._get_full_path(mode), mode)
        else:
            return LocalEmbedder(self._get_full_path(mode), mode)

    def load(self, cube_path):
        with open(cube_path, 'r') as f:
            cube_params = json.loads(f.read())

        mode = cube_params["mode"]

        if self.factory_type == FactoryType.LOCAL:
            return LocalEmbedder.load(cube_path, self._get_full_path(mode))
        else:
            return NetworkEmbedder.load(cube_path, self._get_full_url(mode))

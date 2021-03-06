import os
from enum import Enum

from ..embedders import LocalEmbedder, NetworkEmbedder
from ..utils.functions import is_url


class EmbedderFactoryABC(object):
    def create(self, mode):
        raise NotImplementedError


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

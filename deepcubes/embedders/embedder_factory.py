import os
import json

from ..embedders import LocalEmbedder, NetworkEmbedder


class EmbedderFactory(object):

    def __init__(self, local_path=None, network_url=None):
        self.local_path = local_path
        self.network_url = network_url

    def _get_full_url(self, mode):
        return "{}/{}".format(self.network_url, mode)

    def _get_full_path(self, mode):
        return os.path.join(self.local_path, "{}.kv".format(mode))

    def create_network(self, mode):
        return NetworkEmbedder(self._get_full_url(mode), mode)

    def create_local(self, mode):
        return LocalEmbedder(self._get_full_path(mode), mode)

    def load(self, cube_path):
        with open(cube_path, 'r') as f:
            cube_params = json.loads(f.read())

        mode = cube_params["mode"]
        embedder_class_name = cube_params["cube"]

        if self.local_path and embedder_class_name == "LocalEmbedder":
            return LocalEmbedder.load(cube_path, self._get_full_path(mode))

        elif self.network_path and embedder_class_name == "NetworkEmbedder":
            return NetworkEmbedder.load(cube_path, self._get_full_url(mode))

        raise ValueError("Couldn't load embedder {} ({}) with mode {}".format(
            cube_path, embedder_class_name, mode))

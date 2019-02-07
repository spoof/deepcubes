import os
import json

from ..embedders import LocalEmbedder, NetworkEmbedder


class EmbedderFactory(object):

    def __init__(self, local_path=None, network_url=None):
        self.local_path = local_path
        self.network_url = network_url

    def load(self, cube_path):
        with open(cube_path, 'r') as f:
            cube_params = json.loads(f.read())

        mode = cube_params["mode"]
        embedder_class_name = cube_params["cube"]

        if self.local_path and embedder_class_name == "LocalEmbedder":
            return LocalEmbedder.load(
                cube_path,
                os.path.join(self.local_path, "{}.kv".format(mode))
            )

        elif self.network_path and embedder_class_name == "NetworkEmbedder":
            return NetworkEmbedder.load(
                cube_path,
                "{}/{}".format(mode)
            )

        raise ValueError("Couldn't load embedder {} ({}) with mode {}".format(
            cube_path, embedder_class_name, mode))

import requests
import json


class NetworkEmbedder(object):
    """Network embedder"""

    def __init__(self, url):
        self.url = url

    def __call__(self, *input):
        return self.forward(*input)

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

import json
import os

from ..cubes import TrainableCube, PredictorCube
from ..cubes import LogRegClassifier, NetworkEmbedder
from ..cubes import Tokenizer, Pipe


class IntentClassifier(TrainableCube, PredictorCube):

    def __init__(self, embedder_url):
        self.tokenizer = Tokenizer()
        self.embedder = NetworkEmbedder(embedder_url)

        self.vectorizer = Pipe([self.tokenizer, self.embedder])

        self.log_reg_classifier = LogRegClassifier()

    def train(self, intent_labels, intent_phrases,
              embedder_mode, tokenizer_mode):

        self.embedder.train(embedder_mode)
        self.tokenizer.train(tokenizer_mode)

        intent_vectors = [self.vectorizer(phrase)
                          for phrase in intent_phrases]

        self.log_reg_classifier.train(intent_vectors, intent_labels)

    def forward(self, query):
        return self.log_reg_classifier(self.vectorizer(query))

    def save(self, path, name='intent_classifier.cube'):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'tokenizer': self.tokenizer.save(path=path),
            'embedder': self.embedder.save(path=path),
            'log_reg_classifier': self.log_reg_classifier.save(path=path),
        }

        cube_path = os.path.join(path, name)
        with open(cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return cube_path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        model = cls(None)
        model.tokenizer = Tokenizer.load(cube_params['tokenizer'])
        model.embedder = NetworkEmbedder.load(cube_params['embedder'])
        model.vectorizer = Pipe([model.tokenizer, model.embedder])
        model.log_reg_classifier = LogRegClassifier.load(
            cube_params['log_reg_classifier']
        )

        return model

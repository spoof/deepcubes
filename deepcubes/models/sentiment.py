import json
import os

from torch import nn
import torch

from ..cubes import TrainableCube
from ..cubes import Vocabulary

from ..utils.functions import softmax


class SentimentNN(nn.Module):

    def __init__(self, num_tokens, embed_size, hidden_size,
                 num_layers, num_classes):
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(num_tokens, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, hidden=None):
        emb = self.embed(x)

        out, (ht, ct) = self.lstm(emb, hidden)
        out = self.fc(out[:, -1, :])

        return out


class Sentiment(TrainableCube):

    def __init__(self, embed_size, hidden_size):
        self.vocab = Vocabulary()

        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.num_layers = 1
        self.num_classes = 2

        self.classifier = SentimentNN(
            self.vocab.size(),
            self.embed_size,
            self.hidden_size,
            self.num_layers,
            self.num_classes
        )

    def forward(self, query):
        out = self.classifier(torch.LongTensor(
            self.vocab.get_matrix([query])))

        # TODO: think about
        # return proba of first class
        out = out.cpu().data.numpy()[0]
        probas = softmax(out)

        return probas[1]

    def train(self, labels, texts):
        # TODO: implement
        raise NotImplementedError

    def save(self, path, name='intent_classifier.cube'):
        super().save(path, name)

        params_path = os.path.join(path, "nn_params.torch")
        torch.save(self.classifier.state_dict(), params_path)

        cube_params = {
            'cube': self.__class__.__name__,
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size,
            'vocab': self.vocab.save(path=path),
            'nn_params': params_path,
        }

        self.cube_path = os.path.join(path, name)
        with open(self.cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return self.cube_path

    def _reset_classifier(self):
        self.classifier = SentimentNN(
            self.vocab.size(),
            self.embed_size,
            self.hidden_size,
            self.num_layers,
            self.num_classes
        )

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        model = cls(cube_params["embed_size"], cube_params["hidden_size"])
        model.vocab = Vocabulary.load(cube_params["vocab"])
        model._reset_classifier()
        model.classifier.load_state_dict(torch.load(cube_params['nn_params']))

        return model

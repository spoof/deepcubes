import json
import os

from torch import nn

from ..cubes import TrainableCube, PredictorCube
from ..cubes import EditDistanceMatcher


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


class Sentiment(TrainableCube, PredictorCube):

    MAX_EDIT_DISTANCE = 1

    def __init__(self):
        self.classifier = EditDistanceMatcher()

    def forward(self, query):
        return self.classifier(query)

    def train(self, labels, texts):
        self.classifier.train(labels, texts, self.MAX_EDIT_DISTANCE)

    def save(self, path, name='intent_classifier.cube'):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'classifier': self.classifier.save(path=path),
        }

        self.cube_path = os.path.join(path, name)
        with open(self.cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return self.cube_path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        model = cls()
        model.classifier = EditDistanceMatcher.load(cube_params["classifier"])

        return model

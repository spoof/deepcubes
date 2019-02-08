from collections import defaultdict
import pandas as pd
import json
import os

from ..cubes import TrainableCube, PredictorCube, Tokenizer
from ..models import LogisticIntentClassifier


class MultistagIntentClassifier(TrainableCube, PredictorCube):
    """Live dialog model"""

    def __init__(self, major_clf, minor_clf, tokenizer):
        self.major_clf = major_clf
        self.minor_clf = minor_clf

        self.tokenizer = tokenizer

    def train(self, groups_data_path):
        self.groups_data_path = groups_data_path

        # prepare major and minor data
        groups = pd.read_csv(groups_data_path).fillna("")

        self.minor_to_major = dict(zip(groups["minor"], groups["major"]))
        self.minor_to_keywords = dict(zip(
            groups["minor"],
            [set(self.tokenizer(words)) for words in groups["keywords"]]
        ))

        self.minor_to_answer = dict(zip(groups["minor"], groups["answer"]))
        self.major_to_minors = defaultdict(list)
        for major, minor in zip(groups["major"], groups["minor"]):
            self.major_to_minors[major].append(minor)

    def forward(self, query):
        major_intents = self.major_clf(query)
        minor_intents = self.minor_clf(query)

        return self._get_best_intent(query, major_intents, minor_intents)

    def _get_best_intent(self, query, major_intents, minor_intents):
        major = major_intents[0][0]  # top major
        minors = self.major_to_minors[major]

        words = set(self.tokenizer(query))

        # count keywords for each minor
        keywords_counts = dict(zip(
            minors,
            [len(words.intersection(self.minor_to_keywords[minor]))
             for minor in minors]
        ))

        max_count = max(keywords_counts.values())

        # subset best minors (with max keywords counts)
        best_minors = [minor for minor in minors
                       if keywords_counts[minor] == max_count]

        # return top minor intent that contained in best_minors
        for minor_intent in minor_intents:
            if minor_intent[0] in best_minors:
                return self.minor_to_answer[minor_intent[0]]

        # TODO we couldn't pass to this part
        return self.minor_to_answer[minors[0]]

    def save(self, path, name='vera_live_dialog.cube'):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'groups_data_path': self.groups_data_path,
            'major_clf': self.major_clf.save(
                path=os.path.join(path, 'major_clf')
            ),
            'minor_clf': self.minor_clf.save(
                path=os.path.join(path, 'minor_clf')
            ),
            'tokenizer': self.tokenizer.save(path=path),
        }

        self.cube_path = os.path.join(path, name)
        with open(self.cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return self.cube_path

    @classmethod
    def load(cls, path, embedder_factory):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        major_clf = LogisticIntentClassifier.load(cube_params['major_clf'],
                                                  embedder_factory)
        minor_clf = LogisticIntentClassifier.load(cube_params['minor_clf'],
                                                  embedder_factory)
        tokenizer = Tokenizer.load(cube_params['tokenizer'])
        model = cls(major_clf, minor_clf, tokenizer)
        model.train(cube_params['groups_data_path'])
        model.cube_path = path

        return model

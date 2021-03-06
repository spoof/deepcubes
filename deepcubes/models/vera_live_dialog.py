import json
import os

from ..cubes import TrainableCube, PredictorCube
from ..cubes import PatternMatcher
from ..cubes import Max, CubeLabel
from ..cubes import EditDistanceMatcher
from ..models import LogisticIntentClassifier
from ..utils.functions import sorted_labels


from collections import defaultdict


class Generic(TrainableCube, PredictorCube):
    """General algorithms pretrained for specfical tasks"""

    MAX_EDIT_DIST = 1  # may be need to control from user?

    def __init__(self, mode, texts):
        self.mode = mode
        self.texts = texts
        self.labels = []
        self.ed_matcher = EditDistanceMatcher()

    @classmethod
    def create(cls, mode, data_path):
        texts = []
        with open(data_path, "r") as data_file:
            for line in data_file:
                text, line_mode = line.strip().split("\t")
                if line_mode == mode:
                    texts.append(text)

        return cls(mode, texts)

    def train(self, labels):
        self.labels = labels
        self.ed_matcher.train([labels], [self.texts], self.MAX_EDIT_DIST)

    def forward(self, text):
        return self.ed_matcher(text)

    def save(self, path, name='generics.cube'):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'labels': self.labels,
            'mode': self.mode,
            'texts': self.texts,
            'ed_matcher': self.ed_matcher.save(path),
        }

        cube_path = os.path.join(path, name)
        with open(cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return cube_path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        model = cls(cube_params['mode'], cube_params['texts'])
        model.labels = cube_params['labels']
        model.ed_matcher = EditDistanceMatcher.load(cube_params["ed_matcher"])

        return model


class VeraLiveDialog(TrainableCube, PredictorCube):
    """Live dialog model"""

    NOT_UNDERSTAND_PROBA = 0.6  # TODO: recalc according models

    def __init__(self, embedder, generic_data_path):
        self.pattern_matcher = PatternMatcher()

        self.generics = {
            "yes": Generic.create("yes", generic_data_path),
            "no": Generic.create("no", generic_data_path),
            "repeat": Generic.create("repeat", generic_data_path),
            "no_questions": Generic.create("no_questions", generic_data_path)
        }

        self.generic_data_path = generic_data_path
        self.intent_classifier = LogisticIntentClassifier(embedder)

    def train(self, config):
        """Config dictionary

        "tokenizer_mode" (str),
        "labels_settings":
            [
                {
                    "label": (str) label_name,
                    "patterns": [(str)] patterns for PatternMatcher
                    "generics": [(str)] generic names ('yes'/'no'/'repeat')
                    "intent_phrases": [(str)] list with intent phrases
                },
                ...
            ],
        "not_understand_label" (str)

        """

        self.config = config

        pattern_matcher_labels, pattern_matcher_patterns = [], []
        intent_labels, intent_phrases = [], []
        generic_labels = defaultdict(list)

        for data in config["labels_settings"]:
            label = data["label"]

            if "patterns" in data and len(data["patterns"]):
                pattern_matcher_labels.append([label])
                pattern_matcher_patterns.append(data["patterns"])

            if "generics" in data and len(data["generics"]):
                for generic in data["generics"]:
                    generic_labels[generic].append(label)

            if "intent_phrases" in data:
                for phrase in data["intent_phrases"]:
                    intent_labels.append(label)
                    intent_phrases.append(phrase)

        self.pattern_matcher.train(pattern_matcher_labels,
                                   pattern_matcher_patterns)

        for generic in self.generics:
            if generic in generic_labels:
                self.generics[generic].train(generic_labels[generic])
            else:
                self.generics[generic].train([])

        self.intent_classifier.train(intent_labels, intent_phrases,
                                     config["tokenizer_mode"])

    def forward(self, query, labels=[]):
        max = Max([self.intent_classifier, self.pattern_matcher]
                  + list(self.generics.values()))

        answer = [cube_label for cube_label in max(query)
                  if not len(labels) or cube_label.label in labels]

        not_understand_label = CubeLabel(self.config["not_understand_label"],
                                         self.NOT_UNDERSTAND_PROBA)

        return sorted_labels(answer + [not_understand_label])

    def save(self, path, name='vera_live_dialog.cube'):
        super().save(path, name)

        generics_params = {
            name: generic.save(
                path=os.path.join(path, 'generics/{}'.format(name)),
                name='{}_generic.coub'.format(name)
            ) for name, generic in self.generics.items()
        }

        cube_params = {
            'cube': self.__class__.__name__,
            'config': self.config,
            'pattern_matcher': self.pattern_matcher.save(path=path),
            'generics': generics_params,
            'generic_data_path': self.generic_data_path,
            'intent_classifier': self.intent_classifier.save(
                path=os.path.join(path, 'intent_classifier')
            ),
        }

        self.cube_path = os.path.join(path, name)
        with open(self.cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return self.cube_path

    @classmethod
    def load(cls, path, embedder_factory):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        model = cls(None, cube_params["generic_data_path"])

        model.config = cube_params['config']
        model.pattern_matcher = PatternMatcher.load(
            cube_params['pattern_matcher']
        )

        model.generics = {
            name: Generic.load(
                path
            ) for name, path in cube_params['generics'].items()
        }

        model.intent_classifier = LogisticIntentClassifier.load(
            cube_params['intent_classifier'], embedder_factory
        )

        model.cube_path = path
        return model

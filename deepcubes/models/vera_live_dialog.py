import json
import os

from deepcubes.cubes import TrainableCube, PredictorCube
from deepcubes.cubes import PatternMatcher, LogRegClassifier, NetworkEmbedder
from deepcubes.cubes import Tokenizer, Max, Pipe, CubeLabel
from deepcubes.cubes import EditDistanceMatcher

from collections import defaultdict


class Generic(TrainableCube, PredictorCube):
    """General algorithms pretrained for specfical tasks"""

    MAX_EDIT_DIST = 1  # may be need to control from user?

    def __init__(self, tag, data_path):
        self.tag = tag
        self.data_path = data_path
        self.labels = []
        self.ed_matcher = EditDistanceMatcher()

    def train(self, labels):
        self.labels = labels

        texts = []
        with open(self.data_path, "r") as data_file:
            for line in data_file:
                text, tag = line.split("\t")
                if tag == self.tag:
                    texts.append(text)

        self.ed_matcher.train([], [], self.MAX_EDIT_DIST)

    def forward(self, text):
        # TODO: not implemented
        return [CubeLabel(label, 0) for label in self.labels]

    def save(self, path, name='generics.cube'):
        super(Generic, self).save(path, name)
        cube_params = {
            'cube': self.__class__.__name__,
            'labels': self.labels,
            'tag': self.tag,
            'data_path': self.data_path,
        }

        cube_path = os.path.join(path, name)
        with open(cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return cube_path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        model = cls(cube_params['tag'], cube_params['data_path'])
        model.labels = cube_params['labels']

        return model


class IntentClassifier(TrainableCube, PredictorCube):

    def __init__(self, embedder_url):
        self.tokenizer = Tokenizer()
        self.embedder = NetworkEmbedder(embedder_url)

        self.vectorizer = Pipe([self.tokenizer, self.embedder])

        self.log_reg_classifier = LogRegClassifier()

    def train(self, intent_labels, intent_phrases, lang):
        # TODO: correct train options according to lang
        self.tokenizer.train("lem")

        # TODO: lang-tag messy!?
        self.embedder.train(lang)

        intent_vectors = [self.vectorizer(phrase)
                          for phrase in intent_phrases]

        self.log_reg_classifier.train(intent_vectors, intent_labels)

    def forward(self, query):
        return self.log_reg_classifier(self.vectorizer(query))

    def save(self, path, name='intent_classifier.cube'):
        super(IntentClassifier, self).save(path, name)
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


class VeraLiveDialog(TrainableCube, PredictorCube):
    """Live dialog model"""

    def __init__(self, embedder_url, generic_data_path):
        self.pattern_matcher = PatternMatcher()

        self.generics = {
            "yes": Generic("yes", generic_data_path),
            "no": Generic("no", generic_data_path),
            "repeat": Generic("repeat", generic_data_path)
        }

        self.embedder_url = embedder_url
        self.generic_data_path = generic_data_path

        self.intent_classifier = IntentClassifier(embedder_url)

    def train(self, config):
        """Config dictionary

        "lang": 'rus' or 'eng'
        "labels_settings":
            [
                {
                    "label": label_name,
                    "patterns": patterns for PatternMatcher
                    "generics": generic names ('yes'/'no'/'repeat')
                    "intent_phrases": list with intent phrases
                },
                ...
            ]

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

        for generic in generic_labels:
            self.generics[generic].train(generic_labels[generic])

        self.intent_classifier.train(intent_labels, intent_phrases,
                                     config["lang"])

    def forward(self, query, labels=[]):
        max = Max([self.intent_classifier, self.pattern_matcher]
                  + list(self.generics.values()))

        return max(query)

    def save(self, path, name='vera_live_dialog.cube'):
        super(VeraLiveDialog, self).save(path, name)

        generics_params = {
            name: generic.save(
                path=os.path.join(path, 'generics'),
                name='{}_generic.coub'.format(name)
            ) for name, generic in self.generics.items()
        }

        cube_params = {
            'cube': self.__class__.__name__,
            'config': self.config,
            'generics': generics_params,
            'embedder_url': self.embedder_url,
            'generic_data_path': self.generic_data_path,
            'intent_classifier': self.intent_classifier.save(
                path=os.path.join(path, 'intent_classifier')
            ),
        }

        cube_path = os.path.join(path, name)
        with open(cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return cube_path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        model = cls(cube_params["embedder_url"],
                    cube_params["generic_data_path"])

        model.config = cube_params['config']
        model.generics = {
            name: Generic.load(
                path
            ) for name, path in cube_params['generics'].items()
        }

        model.intent_classifier = IntentClassifier.load(
            cube_params['intent_classifier']
        )

        return model

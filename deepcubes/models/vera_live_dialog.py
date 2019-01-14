import json
import os

from deepcubes.cubes import TrainableCube, PredictorCube
from deepcubes.cubes import PatternMatcher, LogRegClassifier, NetworkEmbedder
from deepcubes.cubes import Tokenizer, Max, Pipe, CubeLabel

from collections import defaultdict


class Generic(TrainableCube, PredictorCube):
    """General algorithms pretrained for specfical tasks"""

    def __init__(self):
        self.labels = []

    def train(self, labels):
        self.labels = labels

    def forward(self, *input):
        raise NotImplementedError


class GenericYes(Generic):
    """General yes (agreement) algorithm"""

    def forward(self, text):
        # TODO: not implemented
        return [CubeLabel(label, 0) for label in self.labels]


class GenericNo(Generic):
    """General no (negation) classifier"""

    def forward(self, text):
        # TODO: not implemented
        return [CubeLabel(label, 0) for label in self.labels]


class GenericRepeat(Generic):
    """General repeat asking classifier"""

    def forward(self, text):
        # TODO: not implemented
        return [CubeLabel(label, 0) for label in self.labels]


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


class VeraLiveDialog(TrainableCube, PredictorCube):
    """Live dialog model"""

    def __init__(self, embedder_url):
        self.pattern_matcher = PatternMatcher()
        self.generics = {
            "yes": GenericYes(),
            "no": GenericNo(),
            "repeat": GenericRepeat()
        }
        self.intent_classifier = IntentClassifier(embedder_url)

    def train(self, config):
        """Config dictionary

        "mode": 'lem' or 'token'
        "lang": 'rus' or 'eng'
        "emb_url": url for NetworkEmbedder
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
                pattern_matcher_labels.append(label)
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

    def save(self, name='vera_live_dialog.cube', path='scripts/models'):
        new_model_id = self._get_new_model_id(path)
        model_path = os.path.join(path, str(new_model_id))
        os.makedirs(model_path, exist_ok=True)
        cube_params = {
            'cube': self.__class__.__name__,
            'config': self.config,
            'tokenizer': self.tokenizer.save(path=model_path),
            'embedder': self.embedder.save(path=model_path),
            'intent_classifier': self.intent_classifier.save(path=model_path),
            # TODO Add pattern_matcher saving
        }

        cube_path = os.path.join(path, name)
        with open(cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return cube_path

    def _get_new_model_id(self, path):
        model_ids = sorted([model_id for model_id in os.listdir(path)])
        new_model_id = model_ids[-1] + 1 if len(model_ids) else 0
        return new_model_id

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        model = cls()
        model.config = cube_params['config']
        if cube_params['tokenizer']:
            model.tokenizer = Tokenizer.load(cube_params['tokenizer'])
        if cube_params['embedder']:
            model.embedder = NetworkEmbedder.load(cube_params['embedder'])
        if cube_params['intent_classifier']:
            model.intent_classifier = LogRegClassifier.load(
                cube_params['intent_classifier']
            )
        # TODO Add pattern_matcher loading
        return model

from deepcubes.models import Model
from deepcubes.cubes import Cube
from deepcubes.cubes import PatternMatcher, LogRegClassifier, NetworkEmbedder


class VeraGeneric(Cube):
    pass


class VeraGenericYes(VeraGeneric):
    pass


class VeraGenericNo(VeraGeneric):
    pass


class VeraGenericRepeat(VeraGeneric):
    pass


class VeraLiveDialog(Model):
    """Live dialog model"""

    def __init__(self):
        self.embedder = NetworkEmbedder()
        self.pattern_matcher = PatternMatcher()
        self.intent_classifier = LogRegClassifier()

    def _construct_generics(self):
        self.generics = {
            "yes": VeraGenericYes(),
            "no": VeraGenericNo()
        }

    def train(self, config):
        """Config dictionary

        "lang": 'rus' or 'eng'
        "labels_settings":
            [
                {
                    "label": label_name,
                    "patterns": patterns for PatternMatcher
                    "generics": generic names
                    "intent_phrases": list with intent phrases
                },
                ...
            ]

        """

        self._construct_generics()

        # TODO: self.embedder.train(config["lang"])

        pattern_matcher_labels, pattern_matcher_patterns = [], []
        intent_labels, intent_phrases = [], []

        for data in config["labels_settings"]:
            label = data["label"]

            if "patterns" not in data or not len(data["patterns"]):
                continue

            pattern_matcher_labels.append(label)
            pattern_matcher_patterns.append(data["patterns"])

        self.pattern_matcher.train(pattern_matcher_labels,
                                   pattern_matcher_patterns)

    def predict(self, query, labels=[]):
        pattern_matcher_results = self.pattern_matcher(query)
        return pattern_matcher_results

from deepcubes.models import Model
from deepcubes.cubes import PatternMatcher


class VeraLiveDialog(Model):
    """Live dialog model"""

    def __init__(self):
        self.pattern_matcher = PatternMatcher()

    def train(self, config):
        """Config dictionary

        List of dicts:
            {
                "label": label_name,
                "patterns": patterns for PatternMatcher
                "generics": generic names
                "intent_phrases": list with intent phrases
            }

        """

        pattern_matcher_labels = []
        pattern_matcher_patterns = []

        for data in config:
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

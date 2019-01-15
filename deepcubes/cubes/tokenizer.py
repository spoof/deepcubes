from .cube import TrainableCube

from pymystem3 import Mystem
import string
import json
import os


class Tokenizer(TrainableCube):
    """Word tokenizer"""

    def __init__(self):
        self.mode_dict = {
            'token': self._get_tokenize_words,
            'lem': self._get_lemmitize_words,
            None: self._get_tokenize_words,
        }

        self.stemmer = Mystem()
        self.exclude = set(string.punctuation)

    def _get_lemmitize_words(self, text, letter_limit=3):
        """Lemmtize text using mystem stemmer"""

        processed = self.stemmer.analyze(text)

        lemmas = []
        for word in processed:
            if "analysis" not in word or not len(word["analysis"]):
                continue

            if len(word['text']) <= letter_limit:
                continue

            analysis = word["analysis"][0]
            if "lex" not in analysis or "gr" not in analysis:
                continue

            lemma = analysis["lex"].lower().strip()
            tag = analysis["gr"].split(",")[0].split("=")[0]

            lemmas.append("{}_{}".format(lemma, tag))

        return lemmas

    def _get_tokenize_words(self, text, letter_limit=2):
        """Delete words with fewer than `n` letters"""
        words = [word.lower() for word in text.split()
                 if len(word) > letter_limit]

        return words

    def _text_clean(self, text):
        text_cleared = ''.join(ch for ch in text if ch not in self.exclude)
        text_cleared = text_cleared.lower()

        return text_cleared

    def train(self, mode=None):
        self.mode = mode if mode in self.mode_dict else None

    def forward(self, text):
        tokenizer = self.mode_dict[self.mode]
        clean_text = self._text_clean(text)
        tokens = tokenizer(clean_text)

        return tokens

    def save(self, path, name='token.cube'):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'mode': self.mode
        }

        cube_path = os.path.join(path, name)

        with open(cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return cube_path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        tokenizer = cls()
        tokenizer.train(cube_params["mode"])

        return tokenizer

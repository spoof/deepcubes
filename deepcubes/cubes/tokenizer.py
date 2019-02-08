from .cube import TrainableCube
from pymystem3 import Mystem
import string
import json
import os
import re


class Tokenizer(TrainableCube):
    """Word tokenizer"""

    def __init__(self):
        self.modes = {
            'token': self._get_tokenize_words,
            'btoken': self._get_btokenize_words,
            'lem': self._get_lemmitize_words,
            None: self._get_tokenize_words,
        }

        self.stemmer = Mystem()
        self.exclude = set(string.punctuation)

    def _get_lemmitize_words(self, text):
        """Lemmtize text using mystem stemmer"""

        processed = self.stemmer.analyze(text)

        lemmas = []
        for word in processed:
            if "analysis" not in word or not len(word["analysis"]):
                continue

            if len(word['text']) <= self.letter_limit:
                continue

            analysis = word["analysis"][0]
            if "lex" not in analysis or "gr" not in analysis:
                continue

            lemma = analysis["lex"].lower().strip()
            tag = analysis["gr"].split(",")[0].split("=")[0]

            lemmas.append("{}_{}".format(lemma, tag))

        return lemmas

    def _get_tokenize_words(self, text):
        """Delete words with fewer than `n` letters"""

        words = [word.lower() for word in text.split()
                 if not len(word) <= self.letter_limit]

        return words

    def _get_btokenize_words(self, text):
        lower_text = text.lower()
        cleanded_text = re.sub("\\(.*?\\)", "", lower_text)

        tokens = self.stemmer.analyze(cleanded_text)
        lemmas = []
        for token in tokens:

            if "analysis" in token:
                if len(token["analysis"]):
                    if "lex" in token["analysis"][0]:
                        lemmas.append(token["analysis"][0]["lex"])
                else:
                    if "text" not in token:
                        continue

                    # english
                    if re.match("[a-z]", token["text"].lower()):
                        if not len(lemmas) or lemmas[-1] != "_ENG_":
                            lemmas.append("_ENG_")
                    else:
                        if re.sub(r"\s", "", token["text"]):
                            if not len(lemmas) or lemmas[-1] != "_NAME_":
                                lemmas.append("_NAME_")
            else:
                if "text" not in token:
                    continue

                if token["text"].isdigit():
                    if not len(lemmas) or lemmas[-1] != "_DIGITS_":
                        lemmas.append("_DIGITS_")
                else:
                    clean_text = re.sub(" ", "", token["text"])
                    if clean_text in [".", ",", "?", "!", ":", "(", ")"]:
                        lemmas.append(clean_text)

        return lemmas

    def _text_clean(self, text):
        text_cleared = ''.join(ch for ch in text if ch not in self.exclude)
        text_cleared = text_cleared.lower()
        return text_cleared

    def train(self, mode, letter_limit=3):
        self.mode = mode if mode in self.modes else None
        self.letter_limit = letter_limit

    def forward(self, text):
        cleaned_text = self._text_clean(text)

        tokenizer = self.modes[self.mode]
        tokens = tokenizer(cleaned_text)

        return tokens

    def save(self, path, name='token.cube'):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'mode': self.mode,
            'letter_limit': self.letter_limit,
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
        tokenizer.train(cube_params['mode'], cube_params['letter_limit'])

        return tokenizer

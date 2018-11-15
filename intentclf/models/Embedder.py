from gensim.models import KeyedVectors
import numpy as np
from pymystem3 import Mystem


class Embedder(object):
    """Word embedder"""

    def __init__(self, path, language=None):
        self.path = path
        self.model = KeyedVectors.load(path, mmap='r')
        self.stemmer = Mystem()
        self.language = language

    def _get_lemmitize_words(self, text):
        """Lemmtize text using mystem stemmer"""
        processed = self.stemmer.analyze(text)

        lemmas = []
        for word in processed:
            if "analysis" not in word or not len(word["analysis"]):
                continue
            if len(word['text']) <= 3:
                continue
            analysis = word["analysis"][0]
            if "lex" not in analysis or "gr" not in analysis:
                continue

            lemma = analysis["lex"].lower().strip()
            tag = analysis["gr"].split(",")[0].split("=")[0]

            lemmas.append("{}_{}".format(lemma, tag))

        return lemmas

    def _get_simple_lemmitize_words(self, text):
        words = [word.lower() for word in text.split() if len(word) > 2]
        return words

    def get_vector(self, text):
        """Calculate vector for sentence as mean vector of all its words"""
        if self.language == 'ru':
            words = self._get_lemmitize_words(text)
        else:
            words = self._get_simple_lemmitize_words(text)

        vector = np.zeros(self.model.vector_size)
        words_in_model = 0

        for word in words:
            if word in self.model:
                vector += self.model.get_vector(word)
                words_in_model += 1

        if words_in_model:
            vector /= words_in_model

        return vector

from .cube import TrainableCube

from collections import defaultdict
import pickle
import tqdm
import numpy as np


class Vocabulary(TrainableCube):

    def __init__(self, max_words=None, min_count=None):
        self.max_words = max_words
        self.min_count = min_count
        self.ids = {}

        self._reset_ids()

    def _reset_ids(self):
        self.ids = {
            "_SOS_": 0,
            "_EOS_": 1,
            "_UNK_": 2,
            "_PAD_": 3
        }

    def save(self, path):
        with open(path, "wb") as outfile:
            pickle.dump({
                "max_words": self.max_words,
                "min_count": self.min_count,
                "ids": self.ids
            }, outfile)

    def load(self, path):
        with open(path, "rb") as infile:
            data = pickle.load(infile)

            self.max_words = data["max_words"]
            self.min_count = data["min_count"]
            self.ids = data["ids"]

    def fit(self, texts):
        words_counts = defaultdict(int)

        for text in tqdm.tqdm(texts):
            for word in text.split(" "):
                words_counts[word] += 1

        self._reset_ids()
        words_stat = sorted(words_counts.items(), key=lambda k: -k[1])
        for it, (word, count) in enumerate(words_stat):

            if self.max_words and it > self.max_words:
                break

            if self.min_count and count < self.min_count:
                break

            self.ids[word] = len(self.ids)

    def size(self):
        return len(self.ids)

    def get_id(self, word):
        if word in self.ids:
            return self.ids[word]
        else:
            return self.ids["_UNK_"]

    def get_matrix(self, texts, max_len=None):
        words = [text.split() for text in texts]

        mtx_len = max([len(w) for w in words])
        if max_len and max_len < mtx_len:
            mtx_len = max_len

        mtx = np.zeros((len(texts), mtx_len + 2)) + self.ids["_PAD_"]

        for ix, text_words in enumerate(words):
            mtx[ix, 0] = self.ids["_SOS_"]

            for iy, word in enumerate(text_words[:mtx_len]):
                mtx[ix, iy + 1] = self.get_id(word)

            mtx[ix, min(mtx_len, len(text_words)) + 1] = self.ids["_EOS_"]

        return mtx

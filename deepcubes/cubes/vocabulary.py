from .cube import TrainableCube

from collections import defaultdict
import tqdm
import numpy as np
import os
import json


class Vocabulary(TrainableCube):

    def __init__(self, max_words=None, min_count=None):
        self.max_words = max_words
        self.min_count = min_count
        self.ids = {}

        # TODO: switch to correct tokenizer class
        self.tokenizer = lambda x: x.lower().split()

        self._reset_ids()

    def _reset_ids(self):
        self.ids = {
            "_SOS_": 0,
            "_EOS_": 1,
            "_UNK_": 2,
            "_PAD_": 3
        }

    def save(self, path, name='vocabulary.cube'):
        super().save(path, name)

        cube_params = {
            'cube': self.__class__.__name__,
            'max_words': self.max_words,
            'min_count': self.min_count,
            'ids': self.ids
        }

        cube_path = os.path.join(path, name)
        with open(cube_path, 'w') as out:
            out.write(json.dumps(cube_params))

        return cube_path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cube_params = json.loads(f.read())

        vocab = cls(cube_params["max_words"], cube_params["min_count"])
        vocab.ids = cube_params["ids"]

        return vocab

    def train(self, texts):
        words_counts = defaultdict(int)

        for text in tqdm.tqdm(texts):
            for word in self.tokenizer(text):
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
        words = [self.tokenizer(text) for text in texts]

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

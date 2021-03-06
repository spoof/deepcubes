import unittest
import numpy as np
import os

from deepcubes.cubes import Tokenizer
from deepcubes.embedders import LocalEmbedder


class TestEmbedder(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer()
        self.emb_path = 'tests/data/test_embeds.kv'
        self.embedder = LocalEmbedder(self.emb_path)
        self.text_phrase = ' '.join([
            'компании',
            'Робот',
            'какая',
            'называетесь',
            'нейросетей',
            'Ваша',
        ])

        self.data_dir = 'tests/data'

    def test_tokenizer(self):
        self.tokenizer.train('lem')

        lem_result = [
            'компания_S',
            'робот_S',
            'какой_APRO',
            'называться_V',
            'нейросеть_S',
            'ваш_APRO',
            ]

        self.assertEqual(self.tokenizer(self.text_phrase), lem_result)

    def test_tokenizer_loading(self):
        name, mode = 'token.cube', 'tokens'

        self.tokenizer.train(mode=mode)
        self.tokenizer.save(name=name, path=self.data_dir)

        new_tokenizer = Tokenizer.load(path=os.path.join(self.data_dir, name))
        self.assertEqual(self.tokenizer.mode, new_tokenizer.mode)

        os.remove(os.path.join(self.data_dir, name))

    def test_get_zero_vector(self):
        np.testing.assert_almost_equal(self.embedder([]),
                                       np.array([0.0]*100), 1)

        np.testing.assert_almost_equal(self.embedder(['test']),
                                       np.array([0.0]*100), 1)

    def test_get_vector(self):
        self.tokenizer.train('lem')

        tokens = self.tokenizer('Робот Вера')
        generated_vector = self.embedder(tokens)

        np.testing.assert_almost_equal(
            sum(generated_vector),
            -0.030646920857179794,
            1
        )

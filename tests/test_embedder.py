import unittest
import numpy as np
import os

from deepcubes.cubes import Tokenizer, Embedder


class TestEmbedder(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer()
        self.embedder = Embedder()
        self.text_phrase = ' '.join([
            'компании',
            'Робот',
            'какая',
            'называетесь',
            'нейросетей',
            'Ваша',
        ])
        self.data_dir = 'tests/data'
        self.emb_path = 'tests/data/test_embeds.kv'
        self.embedder.train(self.emb_path)

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
        name = 'token.cube'
        mode = 'tokens'
        self.tokenizer.train(mode=mode)
        self.tokenizer.save(name=name, path=self.data_dir)
        new_tokenizer = Tokenizer.load(path=os.path.join(self.data_dir, name))
        self.assertEqual(self.tokenizer.mode, new_tokenizer.mode)
        os.remove(os.path.join(self.data_dir, name))

    def test_get_zero_vector(self):
        np.testing.assert_almost_equal(
            self.embedder([]), np.array([0.0, 0.0]), 1
        )
        np.testing.assert_almost_equal(
            self.embedder(['test']), np.array([0.0, 0.0]), 1
        )

    def test_get_vector(self):
        self.tokenizer.train('lem')
        tokens = self.tokenizer('Робот Вера')
        generated_vector = self.embedder(tokens)
        correct_vector = np.array([0.5, 0.6])
        np.testing.assert_almost_equal(generated_vector, correct_vector, 1)

    def test_embedder_loading(self):
        name = 'embedder.cube'
        self.embedder.save(name=name, path=self.data_dir)
        new_embedder = Embedder.load(path=os.path.join(self.data_dir, name))
        self.assertEqual(self.embedder.path, new_embedder.path)
        os.remove(os.path.join(self.data_dir, name))

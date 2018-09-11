import unittest
import numpy as np

from intentclf.models import Embedder


class TestEmbedder(unittest.TestCase):

    def setUp(self):
        self.embedder = Embedder('tests/data/test_embeds.kv')

    def test_get_vector(self):
        generated_vector = self.embedder.get_vector('Робот Вера')
        correct_vector = np.array([0.5, 0.6])
        np.testing.assert_almost_equal(generated_vector, correct_vector, 1)

    def test_get_lemmitize_words(self):
        words = [
            'компании',
            'Робот',
            'какая',
            'называетесь',
            'нейросетей',
            'ваша',
            ]
        words = ' '.join(words)
        result = [
            'компания_S',
            'робот_S',
            'какой_APRO',
            'называться_V',
            'нейросеть_S',
            'ваш_APRO',
            ]

        self.assertEqual(self.embedder._get_lemmitize_words(words), result)

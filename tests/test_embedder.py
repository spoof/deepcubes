import unittest
import numpy as np

from intentclf.models import Embedder


class TestEmbedder(unittest.TestCase):

    def setUp(self):
        self.embedder = Embedder('tests/data/test_embeds.kv')

    def test_get_vector(self):
        text = 'Робот Вера'

        v1 = self.embedder.get_vector(text)
        v1 = np.vectorize(lambda x: round(x, 1))(v1)

        v2 = np.array([0.0, 0.0])
        for word in text.split():
            v2 += np.array(self.embedder.get_vector(word))
        v2 = v2/len(text.split())
        v2 = np.vectorize(lambda x: round(x, 1))(v2)

        self.assertEqual(v1.tolist(), v2.tolist())

    def test_get_lemmitize_words(self):
        words = ['компании',
                 'Робот',
                 'какая',
                 'называетесь',
                 'нейросетей',
                 'ваша',
                 ]
        words = ' '.join(words)
        result = ['компания_S',
                  'робот_S',
                  'какой_APRO',
                  'называться_V',
                  'нейросеть_S',
                  'ваш_APRO',
                  ]

        self.assertEqual(self.embedder._get_lemmitize_words(words), result)

import unittest
from deepcubes.cubes import Vocabulary
import numpy as np


class TestVocabulary(unittest.TestCase):

    def test_vocabulary(self):
        vocab = Vocabulary()
        vocab.train(["hello", "hello yes", "hello"])

        self.assertEqual(vocab.size(), 6)

        np.testing.assert_almost_equal(
            vocab.get_matrix(["hello"]),
            [[0, 4, 1]]
        )

        np.testing.assert_almost_equal(
            vocab.get_matrix(["hello", "hello hello yes"]),
            [
                [0, 4, 1, 3, 3],
                [0, 4, 4, 5, 1]
            ]
        )

        np.testing.assert_almost_equal(
            vocab.get_matrix(["hello", "hello hello yes"], max_len=2),
            [
                [0, 4, 1, 3],
                [0, 4, 4, 1]
            ]
        )

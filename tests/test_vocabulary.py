import unittest
from deepcubes.cubes import Vocabulary


class TestVocabulary(unittest.TestCase):

    def test_vocabulary(self):
        vocab = Vocabulary()
        vocab.fit(["hello", "yes"])

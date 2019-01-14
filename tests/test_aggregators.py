import unittest
from deepcubes.cubes import Max, PatternMatcher, EditDistanceMatcher, Pipe
from deepcubes.cubes import Embedder, Tokenizer
import numpy as np


class TestAggregators(unittest.TestCase):

    def test_max(self):
        pattern_matcher = PatternMatcher()
        pattern_matcher.train(
            ["first", "third", "second"],
            [
                ["привет", ".*привет.*"],
                ["пока", ".*пока.*", "превед"],
                ["ок"]
            ]
        )

        ed_matcher = EditDistanceMatcher()
        ed_matcher.train(
            ["first", "third", "second"],
            [
                ["пока", "покасики"],
                ["ок"],
                ["привет", "превет"]
            ],
            1
        )

        max_cube = Max([pattern_matcher, ed_matcher])

        self.assertEqual(
            max_cube("привет"),
            [("first", 1), ("third", 0), ("second", 1)]
        )

        self.assertEqual(
            max_cube("превет"),
            [("first", 0), ("third", 0), ("second", 1)]
        )

        self.assertEqual(
            max_cube("покасики"),
            [("first", 1), ("third", 1), ("second", 0)]
        )

    def test_pipe(self):

        tokenizer = Tokenizer()
        tokenizer.train('lem')

        embedder = Embedder()
        emb_path = 'tests/data/test_embeds.kv'
        embedder.train(emb_path)

        pipe = Pipe([tokenizer, embedder])

        np.testing.assert_almost_equal(
            pipe("Робот Вера"),
            [0.5, 0.6],
            1
        )
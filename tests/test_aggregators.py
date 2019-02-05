import unittest
from deepcubes.cubes import Max, PatternMatcher, EditDistanceMatcher, Pipe
from deepcubes.cubes import Tokenizer
from deepcubes.embedders import Embedder
import numpy as np


class TestAggregators(unittest.TestCase):

    def test_max(self):
        pattern_matcher = PatternMatcher()
        pattern_matcher.train(
            [["first"], ["third"], ["second"]],
            [
                ["привет", ".*привет.*"],
                ["пока", ".*пока.*", "превед"],
                ["ок"]
            ]
        )

        ed_matcher = EditDistanceMatcher()
        ed_matcher.train(
            [["first"], ["third"], ["second"]],
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
            [("first", 1), ("second", 1), ("third", 0)]
        )

        self.assertEqual(
            max_cube("превет"),
            [("second", 1), ("first", 0), ("third", 0)]
        )

        self.assertEqual(
            max_cube("покасики"),
            [("first", 1), ("third", 1), ("second", 0)]
        )

    def test_pipe(self):

        tokenizer = Tokenizer()
        tokenizer.train('lem')

        emb_path = 'tests/data/test_embeds.kv'
        embedder = Embedder(emb_path)

        pipe = Pipe([tokenizer, embedder])

        np.testing.assert_almost_equal(
            sum(pipe("Робот Вера")),
            -0.030646920857179794,
            1
        )

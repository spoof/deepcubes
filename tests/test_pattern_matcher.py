import unittest
from deepcubes.cubes import PatternMatcher


class TestPatternMatcher(unittest.TestCase):

    def test_matcher(self):
        matcher = PatternMatcher()
        matcher.train(
            [["first"], ["third"], ["second"]],
            [
                ["привет", ".*привет.*"],
                ["пока", ".*пока.*", "превед"],
                ["ок"]
            ]
        )

        self.assertEqual(
            matcher("привет"),
            [("first", 1), ("second", 0), ("third", 0)]
        )

        self.assertEqual(
            matcher("как ты, привет"),
            [("first", 1), ("second", 0), ("third", 0)]
        )

        self.assertEqual(
            matcher("как ты, превет"),
            [("first", 0), ("second", 0), ("third", 0)]
        )

        self.assertEqual(
            matcher("как ты, пока и привет"),
            [("first", 1), ("third", 1), ("second", 0)]
        )

        self.assertEqual(
            matcher("ок"),
            [("second", 1), ("first", 0), ("third", 0)]
        )

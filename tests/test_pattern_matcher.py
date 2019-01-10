import unittest
from deepcubes.cubes import PatternMatcher


class TestPatternMatcher(unittest.TestCase):

    def test_matcher(self):
        matcher = PatternMatcher()
        matcher.train(
            ["first", "third", "second"],
            [
                ["привет", ".*привет.*"],
                ["пока", ".*пока.*", "превед"],
                ["ок"]
            ]
        )

        self.assertEqual(
            matcher("привет"),
            (["first", "third", "second"], [1, 0, 0])
        )

        self.assertEqual(
            matcher("как ты, привет"),
            (["first", "third", "second"], [1, 0, 0])
        )

        self.assertEqual(
            matcher("как ты, превет"),
            (["first", "third", "second"], [0, 0, 0])
        )

        self.assertEqual(
            matcher("как ты, пока и привет"),
            (["first", "third", "second"], [1, 1, 0])
        )

        self.assertEqual(
            matcher("ок"),
            (["first", "third", "second"], [0, 0, 1])
        )

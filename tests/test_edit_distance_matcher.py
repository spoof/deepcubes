import unittest
from deepcubes.cubes import EditDistanceMatcher


class TestEditDistanceMatcher(unittest.TestCase):

    def test_matcher(self):
        matcher = EditDistanceMatcher()
        matcher.train(
            ["first", "third", "second"],
            [
                ["привет", "превет"],
                ["пока", "покасики", "превед"],
                ["ок"]
            ],
            1
        )

        self.assertEqual(
            matcher("привет"),
            [("first", 1), ("third", 0), ("second", 0)]
        )

        self.assertEqual(
            matcher("прувет"),
            [("first", 1), ("third", 0), ("second", 0)]
        )

        self.assertEqual(
            matcher("пруветик"),
            [("first", 0), ("third", 0), ("second", 0)]
        )

        self.assertEqual(
            matcher("пруветик"),
            [("first", 0), ("third", 0), ("second", 0)]
        )

        self.assertEqual(
            matcher("привед"),
            [("first", 1), ("third", 1), ("second", 0)]
        )

        self.assertEqual(
            matcher("оке"),
            [("first", 0), ("third", 0), ("second", 1)]
        )

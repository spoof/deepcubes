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
            matcher.predict("привет"),
            (["first", "third", "second"], [1, 0, 0])
        )

        self.assertEqual(
            matcher.predict("прувет"),
            (["first", "third", "second"], [1, 0, 0])
        )

        self.assertEqual(
            matcher.predict("пруветик"),
            (["first", "third", "second"], [0, 0, 0])
        )

        self.assertEqual(
            matcher.predict("пруветик"),
            (["first", "third", "second"], [0, 0, 0])
        )

        self.assertEqual(
            matcher.predict("привед"),
            (["first", "third", "second"], [1, 1, 0])
        )

        self.assertEqual(
            matcher.predict("оке"),
            (["first", "third", "second"], [0, 0, 1])
        )

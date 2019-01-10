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
            (["first", "third", "second"], [1, 0, 0])
        )

        self.assertEqual(
            matcher("прувет"),
            (["first", "third", "second"], [1, 0, 0])
        )

        self.assertEqual(
            matcher("пруветик"),
            (["first", "third", "second"], [0, 0, 0])
        )

        self.assertEqual(
            matcher("пруветик"),
            (["first", "third", "second"], [0, 0, 0])
        )

        self.assertEqual(
            matcher("привед"),
            (["first", "third", "second"], [1, 1, 0])
        )

        self.assertEqual(
            matcher("оке"),
            (["first", "third", "second"], [0, 0, 1])
        )

import unittest
from deepcubes.models import VeraLiveDialog


class TestVeraLiveDialog(unittest.TestCase):

    def test_patterns(self):
        vera = VeraLiveDialog()
        vera.train({
            "labels_settings": [
                {
                    "label": "hello",
                    "patterns": ["привет", ".*привет.*"]
                },
                {
                    "label": "bye-bye",
                    "patterns": ["пока", "пока-пока.*"]
                }]
        })

        self.assertEqual(
            vera("привет"),
            [("hello", 1), ("bye-bye", 0)]
        )

        self.assertEqual(
            vera("приветик"),
            [("hello", 1), ("bye-bye", 0)]
        )

        self.assertEqual(
            vera("прив пока-пока"),
            [("hello", 0), ("bye-bye", 0)]
        )

        self.assertEqual(
            vera("пока-пока привет"),
            [("hello", 1), ("bye-bye", 1)]
        )

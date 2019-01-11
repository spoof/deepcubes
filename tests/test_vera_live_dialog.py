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
            (["hello", "bye-bye"], [1, 0])
        )

        self.assertEqual(
            vera("приветик"),
            (["hello", "bye-bye"], [1, 0])
        )

        self.assertEqual(
            vera("прив пока-пока"),
            (["hello", "bye-bye"], [0, 0])
        )

        self.assertEqual(
            vera("пока-пока привет"),
            (["hello", "bye-bye"], [1, 1])
        )

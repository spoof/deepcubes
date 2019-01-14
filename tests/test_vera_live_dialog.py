import unittest
from deepcubes.models import VeraLiveDialog


class TestVeraLiveDialog(unittest.TestCase):

    def test_vera_dialog(self):
        # TODO: get path from environment
        vera = VeraLiveDialog("http://51.144.105.1:3349/get_vector")

        vera.train({
            "lang": "test",
            "labels_settings": [
                {
                    "label": "hello",
                    "patterns": ["привет", ".*привет.*"],
                    "intent_phrases": ["привет", "хай"]
                },
                {
                    "label": "bye-bye",
                    "patterns": ["пока", "пока-пока.*"],
                    "intent_phrases": ["пока", "прощай"]
                }]
        })

        # TODO: near equal checking
        self.assertEqual(
            vera("привет"),
            [("hello", 1), ("bye-bye", 0.5)]
        )

        # TODO: near equal checking
        self.assertEqual(
            vera("приветик"),
            [("hello", 1), ("bye-bye", 0.5)]
        )

        self.assertEqual(
            vera("прив пока-пока"),
            [("hello", 0.5), ("bye-bye", 0.5)]
        )

        self.assertEqual(
            vera("пока-пока привет"),
            [("hello", 1), ("bye-bye", 1)]
        )

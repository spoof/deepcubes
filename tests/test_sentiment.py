import unittest
from deepcubes.models import Sentiment


class TestSentiment(unittest.TestCase):

    def test_sentiment(self):
        sentiment = Sentiment()

        sentiment.train(
            [["positive"], ["negative"]],
            [["классно", "супер"], ["плохо", "не нравится"]]
        )

        self.assertEqual(
            sentiment("бомба"),
            [("negative", 0), ("positive", 0)]
        )

        self.assertEqual(
            sentiment("не нравиться"),
            [("negative", 1), ("positive", 0)]
        )

        self.assertEqual(
            sentiment("не нравится"),
            [("negative", 1), ("positive", 0)]
        )

        self.assertEqual(
            sentiment("класс"),
            [("negative", 0), ("positive", 0)]
        )

        self.assertEqual(
            sentiment("классн"),
            [("positive", 1), ("negative", 0)]
        )

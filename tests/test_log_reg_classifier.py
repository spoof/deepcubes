import unittest
import json

from deepcubes.cubes import Tokenizer, LogRegClassifier
from deepcubes.embedders import LocalEmbedder


class TestLogRegClassifier(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer()
        self.embedder = LocalEmbedder('tests/data/test_embeds.kv')
        self.classifier = LogRegClassifier()

        self.tokenizer.train('lem')

        self.questions, self.answers = [], []

        with open("tests/data/test_dialog.json", "r") as handle:
            data = json.load(handle)

        for label, category in enumerate(data):
            answer = category["answers"][0]

            for question in category["questions"]:
                self.questions.append(question)
                self.answers.append(answer)

        self.answer_to_label = dict()
        self.label_to_answer = dict()
        self.X, self.Y = [], []

        idx = 0
        for answer, question in zip(self.answers, self.questions):
            if answer not in self.answer_to_label:
                self.answer_to_label[answer] = idx
                self.label_to_answer[idx] = answer
                idx += 1

            vector = self.embedder(self.tokenizer(question))
            self.Y.append(self.answer_to_label[answer])
            self.X.append(vector)

    def test_extreme_cases(self):
        clf = LogRegClassifier()

        clf.train([], [])
        self.assertEqual(clf([[0, 1, 2]]), [])

        clf.train([[1, 2, 3]], [1])
        self.assertEqual(clf([0, 0, 0]), [(1, 1.0)])
        self.assertEqual(clf([0, 1, 2]), [(1, 1.0)])

    def test_saving_and_loading(self):
        self.classifier.train(self.X, self.Y)

        clf = self.classifier.clf

        clf_params = self.classifier.save()
        new_classifier = LogRegClassifier.load(clf_params)

        self.assertEqual(clf.coef_.tolist(),
                         new_classifier.clf.coef_.tolist())

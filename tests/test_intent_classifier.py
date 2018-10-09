import unittest
import os
import json
import sklearn

from intentclf.models import Embedder
from intentclf.models import IntentClassifier


class TestIntentClassifier(unittest.TestCase):

    def setUp(self):
        self.embedder = Embedder('tests/data/test_embeds.kv')
        self.classifier = IntentClassifier(self.embedder)

        self.questions, self.answers = [], []

        with open("tests/data/test_dialog.json", "r") as handle:
            data = json.load(handle)

        for label, category in enumerate(data):
            answer = category["answers"][0]

            for question in category["questions"]:
                self.questions.append(question)
                self.answers.append(answer)

        self.checkpoint_path = 'tests/data/'

    def test_environment_variable(self):
        self.assertIn('INTENT_CLASSIFIER_MODEL', os.environ)

    def test_train_method(self):
        self.classifier.train(self.questions, self.answers)

        answer_to_label = tuple(self.classifier.answer_to_label.items())
        label_to_answer = tuple(self.classifier.label_to_answer.items())
        reversed_label_to_answer = tuple(map(lambda x: (x[1], x[0]),
                                             label_to_answer))

        self.assertEqual(sorted(answer_to_label),
                         sorted(reversed_label_to_answer))

    def test_correct_data_entry(self):
        with self.assertRaises(ValueError):
            self.classifier.train(self.questions[:1], self.answers[:1])

        if (type(self.classifier.clf) is
                sklearn.linear_model.logistic.LogisticRegression):
            with self.assertRaises(sklearn.exceptions.NotFittedError):
                self.classifier.predict('')

    def test_saving_and_loading(self):
        self.classifier.train(self.questions, self.answers)

        clf = self.classifier.clf
        label_to_answer = self.classifier.label_to_answer
        answer_to_label = self.classifier.answer_to_label

        model_id = self.classifier.save(self.checkpoint_path)
        self.classifier.load(model_id, self.checkpoint_path)

        self.assertEqual(clf.coef_.tolist(),
                         self.classifier.clf.coef_.tolist())
        self.assertEqual(label_to_answer,
                         self.classifier.label_to_answer)
        self.assertEqual(answer_to_label,
                         self.classifier.answer_to_label)

        os.remove(os.path.join(
            self.checkpoint_path, 'model-{}.pickle'.format(model_id))
        )

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
import pickle
import string
import numpy as np
import os


class IntentClassifier(object):
    """Classify"""

    def __init__(self, embedder):
        self.embedder = embedder
        self.clf = LogisticRegression(
            solver="liblinear",
            multi_class="ovr",
        )

        self.label_to_answer = dict()
        self.answer_to_label = dict()
        self.question_to_label = dict()
        self.threshold = None
        self.exclude = set(string.punctuation)

    def train(self, questions, answers):
        """Train classifier at question-answer pairs"""

        X, Y = [], []
        self.label_to_answer = dict()
        self.answer_to_label = dict()
        self.question_to_label = dict()

        for question, answer in zip(questions, answers):
            if answer not in self.answer_to_label:
                new_label = len(self.label_to_answer)
                self.answer_to_label[answer] = new_label
                self.label_to_answer[new_label] = answer

            X.append(self.embedder.get_vector(question))
            Y.append(self.answer_to_label[answer])

            question_cleared = self._text_clean(question)
            self.question_to_label[question_cleared] = self.answer_to_label[
                answer]

        self.clf.fit(X, Y)

    def predict(self, query, exact_match=True):
        query_vector = self.embedder.get_vector(query)
        query_cleared = self._text_clean(query)
        if exact_match:
            if query_cleared in self.question_to_label:
                return (
                    self.label_to_answer[
                        self.question_to_label[query_cleared]
                    ],
                    1.0
                )
        try:
            predict_label = self.clf.predict([query_vector])[0]
            max_probability = np.amax(self.clf.predict_proba([query_vector]))
        except NotFittedError as e:
            # TODO(dima): implement logic
            raise e

        return self.label_to_answer[predict_label], max_probability

    def threshold_calc(
        self,
        trash_questions_path=None,
        itself_percent=0.99,
        trash_percent=0.95,
    ):

        threshold_on_trash = None
        if trash_questions_path:
            trash_probabilities = list()

            with open(trash_questions_path, 'r') as trash_questions:
                for question in trash_questions:
                    _, max_probability = self.predict(question.strip())
                    trash_probabilities.append(max_probability)

            threshold_on_trash = self._get_threshold_value(
                trash_probabilities, trash_percent
            )

        itself_probabilities = list()
        for question in self.question_to_label:
            if (
                self.answer_to_label[self.predict(question)[0]] ==
                self.question_to_label[question]
            ):

                _, max_probability = self.predict(question, exact_match=False)
                itself_probabilities.append(max_probability)

        threshold_on_itself = self._get_threshold_value(
            itself_probabilities, itself_percent, side='right'
        )

        if threshold_on_trash:
            self.threshold = (threshold_on_itself + threshold_on_trash)/2
        else:
            self.threshold = threshold_on_itself

    def _get_threshold_value(self, list_, percent, side='left'):
        if side is 'right':
            percent = 1 - percent
        N = len(list_)
        for idx, value in enumerate(sorted(list_)):
            if idx/N > percent:
                return value
        return value

    def save(self, models_storage_path):
        sorted_models_ids = self._get_sorted_models_ids(models_storage_path)
        if len(sorted_models_ids) == 0:
            new_model_id = 0
        else:
            new_model_id = sorted_models_ids[-1] + 1
        path = os.path.join(
            models_storage_path, 'model-{}.pickle'.format(new_model_id)
        )

        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "model": self.clf,
                    "label_to_answer": self.label_to_answer,
                    "answer_to_label": self.answer_to_label,
                    "question_to_label": self.question_to_label,
                    "threshold": self.threshold,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
                file=handle
            )
        return new_model_id

    def _get_sorted_models_ids(self, path):
        models = [
            file_name for file_name in os.listdir(path) if (
                'pickle' in file_name
            )
        ]
        models_ids = map(
            lambda m: int(m.split('model-')[1].split('.pickle')[0]),
            models
            )
        return sorted(models_ids)

    def load(self, model_id, models_storage_path):
        path = os.path.join(
            models_storage_path, "model-{}.pickle".format(model_id)
        )
        with open(path, "rb") as handle:
            data = pickle.load(handle)

        self.clf = data["model"]
        self.label_to_answer = data["label_to_answer"]
        self.answer_to_label = data["answer_to_label"]
        self.question_to_label = data["question_to_label"]
        self.threshold = data["threshold"]

    def _text_clean(self, text):
        text_cleared = ''.join(ch for ch in text if ch not in self.exclude)
        text_cleared = text_cleared.lower()
        return text_cleared

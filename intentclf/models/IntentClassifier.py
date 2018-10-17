from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold

import pickle
import string
import numpy as np
import os
from collections import defaultdict


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
        self.label_to_accuracy_score = None
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

    def cross_val(self, n_splits=10):
        cross_val_clf = LogisticRegression(
            solver="liblinear",
            multi_class="ovr"
        )
        X, Y = [], []

        for question, label in self.question_to_label.items():
            X.append(self.embedder.get_vector(question))
            Y.append(label)

        cv = KFold(n_splits=n_splits, shuffle=True)

        y_true, y_pred = [], []
        X = np.array(X)
        Y = np.array(Y)

        for train_index, test_index in cv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            cross_val_clf.fit(X_train, Y_train)
            Y_predict = cross_val_clf.predict(X_test)

            y_true.append(list(Y_test))
            y_pred.append(Y_predict)

        y_true_list = list(np.concatenate(y_true))
        y_pred_list = list(np.concatenate(y_pred))

        self.label_to_predict_list = defaultdict(list)

        for label, predict_label in zip(y_true_list, y_pred_list):
            self.label_to_predict_list[label].append(label == predict_label)

        self.label_to_accuracy_score = dict()
        for label in self.label_to_predict_list:
            predictions = self.label_to_predict_list[label]
            self.label_to_accuracy_score[label] = np.mean(predictions)

    def predict(self, query, exact_match=True):
        query_vector = self.embedder.get_vector(query)
        query_cleared = self._text_clean(query)

        if exact_match and query_cleared in self.question_to_label:
            return (
                self.label_to_answer[self.question_to_label[query_cleared]],
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
            self.threshold = (threshold_on_itself + threshold_on_trash) / 2
        else:
            self.threshold = threshold_on_itself

    def _get_threshold_value(self, list_, percent, side='left'):
        if side == 'right':
            percent = 1 - percent

        threshold_idx = int(percent * len(list_))
        threshold_value = sorted(list_)[threshold_idx]

        return threshold_value

    def save(self, models_storage_path):
        new_model_id = self._get_new_model_id(models_storage_path)
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
                    "accuracy_scores": self.label_to_accuracy_score,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
                file=handle
            )

        return new_model_id

    def _get_new_model_id(self, path):
        models = [file_name for file_name in os.listdir(path) if (
            'pickle' in file_name
        )]

        models_ids = map(
            lambda m: int(m.split('model-')[1].split('.pickle')[0]),
            models
        )

        sorted_ids = sorted(models_ids)
        new_model_id = sorted_ids[-1] + 1 if len(sorted_ids) else 0

        return new_model_id

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
        self.threshold = data["threshold"] if "threshold" in data else None
        self.label_to_accuracy_score = data["accuracy_scores"] if (
            "accuracy_scores" in data) else None

    def _text_clean(self, text):
        text_cleared = ''.join(ch for ch in text if ch not in self.exclude)
        text_cleared = text_cleared.lower()
        return text_cleared

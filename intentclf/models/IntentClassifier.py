from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
import pickle


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

    def train(self, questions, answers):
        """Train classifier at question-answer pairs"""

        X, Y = [], []
        self.label_to_answer = dict()
        self.answer_to_label = dict()

        for question, answer in zip(questions, answers):
            if answer not in self.answer_to_label:
                new_label = len(self.label_to_answer)
                self.answer_to_label[answer] = new_label
                self.label_to_answer[new_label] = answer

            X.append(self.embedder.get_vector(question))
            Y.append(self.answer_to_label[answer])

        self.clf.fit(X, Y)

    def predict(self, query):
        query_vector = self.embedder.get_vector(query)

        try:
            predict_label = self.clf.predict([query_vector])[0]
        except NotFittedError as e:
            # TODO(dima): implement logic
            raise e

        return self.label_to_answer[predict_label]

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "model": self.clf,
                    "label_to_answer": self.label_to_answer,
                    "answer_to_label": self.answer_to_label,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
                file=handle
            )

    def load(self, path):
        with open(path, "rb") as handle:
            data = pickle.load(handle)

        self.clf = data["model"]
        self.label_to_answer = data["label_to_answer"]
        self.answer_to_label = data["answer_to_label"]

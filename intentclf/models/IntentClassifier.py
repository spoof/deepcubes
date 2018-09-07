from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
import pickle


class IntentClassifier(object):
    """Classify"""

    def __init__(self, embedder):
        self.embedder = embedder
        self.clf = LogisticRegression()
        self.label_to_answer = []

    def train(self, intents):
        """Train classifier from intents json-dict data"""

        X, Y = [], []

        for label, category in enumerate(intents):
            self.label_to_answer.append(
                category["answers"][0]
            )

            for question in category["questions"]:
                X.append(self.embedder.get_vector(question))
                Y.append(label)

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
                    "label_to_answer": self.label_to_answer
                },
                protocol=pickle.HIGHEST_PROTOCOL,
                file=handle
            )

    def load(self, path):
        with open(path, "rb") as handle:
            data = pickle.load(handle)

        self.clf = data["model"]
        self.label_to_answer = data["label_to_answer"]

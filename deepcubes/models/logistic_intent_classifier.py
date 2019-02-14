from ..cubes import Cube, LogRegClassifier, Pipe, Predictor, Tokenizer, Trainable


class LogisticIntentClassifier(Cube, Trainable, Predictor):

    def __init__(self, embedder):
        self.tokenizer = Tokenizer()
        self.embedder = embedder
        self.vectorizer = Pipe([self.tokenizer, self.embedder])
        self.log_reg_classifier = LogRegClassifier()

    def train(self, intent_labels, intent_phrases, tokenizer_mode):
        self.tokenizer.train(tokenizer_mode)
        intent_vectors = [self.vectorizer(phrase)
                          for phrase in intent_phrases]
        self.log_reg_classifier.train(intent_vectors, intent_labels)

    def forward(self, query):
        return self.log_reg_classifier(self.vectorizer(query))

    def save(self):
        model_params = {
            'class': self.__class__.__name__,
            'tokenizer': self.tokenizer.save(),
            'log_reg_classifier': self.log_reg_classifier.save(),
            'embedder': self.embedder.save()
        }

        return model_params

    @classmethod
    def load(cls, model_params, embedder_factory):

        # get embedder mode
        embedder_params = model_params['embedder']
        embedder = embedder_factory.create(embedder_params["mode"])

        model = LogisticIntentClassifier(embedder)
        model.tokenizer = Tokenizer.load(model_params['tokenizer'])

        model.vectorizer = Pipe([model.tokenizer, model.embedder])
        model.log_reg_classifier = LogRegClassifier.load(
            model_params['log_reg_classifier']
        )

        return model

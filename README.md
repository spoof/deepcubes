# Intent Classifier

Robot Vera implementation of Intent Classifier logic based on combination of word embeddings and additional classifier on the top of them.

### Embeddings

In order to start using the intent classifier it is necessary to load embedding.

`from intentclf.models import Embedder`

`from intentclf.models import IntentClassifier`

`embedder = Embedder(path)`

where `path` - path to trained embedding model (Word2Vec, FastText, GloVe)

`classifier = IntentClassifier(embedder)`

### Train and predict model

To train the model you need to prepare two lists and pass them to the `train()` method

`classifier.train(questions, answers)`

The method `predict_top()` returns a list of `top` classifier answers with probabilities

`classifier.predict_top(question, top)`

### Additional methods
(of the trained model)

`classifier.threshold_calc()` - calculation of threshold probability

`intentclf.cross_val()` - calculation of intents accuracy score

### Loading and saving model

`model_id = classifier.save(models_path)`

`classifier.load(model_id, models_path)`

## Authors

* Dmitry Ischenko
* Yana Balandyuk-Opalinskaya
* Vladimir Sveshnikov
* Svyatoslav Nevyantsev

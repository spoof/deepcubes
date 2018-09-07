from intentclf.models import Embedder
from intentclf.models import IntentClassifier

import json

embedder = Embedder("/mnt/disk/models/glove-hh-embeds.kv")

text = "привет как дела? У меня хорошо"
embedder.get_vector("привет как дела")


classifier = IntentClassifier(embedder)

with open("data/dialog.ru.32.json", "r") as handle:
    data = json.load(handle)

classifier.train(data)
print(classifier.predict("есть ли медицинская страховка?"))

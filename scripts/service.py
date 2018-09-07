from intentclf.models import Embedder
from intentclf.models import IntentClassifier

import json

embedder = Embedder("/mnt/disk/models/glove-hh-embeds.kv")
classifier = IntentClassifier(embedder)

with open("data/dialog.ru.32.json", "r") as handle:
    data = json.load(handle)

classifier.train(data)

questions = [
    "какая у вас зарплата?",
    "можно ли обучаться в компании?",
    "работать по выходным можно?"
]

for question in questions:
    answer = classifier.predict(question)
    print("{} : {}".format(question, answer))

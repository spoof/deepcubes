from intentclf.models import Embedder
from intentclf.models import IntentClassifier

import json
import os

import pandas as pd

embedder = Embedder(os.environ['INTENT_CLASSIFIER_MODEL'])
classifier = IntentClassifier(embedder)


# JSON for live dialog
questions, answers = [], []

# parse data from json
with open("data/dialog.ru.32.json", "r") as handle:
    data = json.load(handle)

for label, category in enumerate(data):
    # answer = category["name"]
    answer = category["answers"][0]

    for question in category["questions"]:
        questions.append(question)
        answers.append(answer)

model_path = "scripts/models/model-1.pickle"

classifier.train(questions, answers)
classifier.save(model_path)

# Data for 7 call-center categories
questions, answers = [], []

# parse from pandas data frame
data = pd.read_csv("data/vera_ml_7cat.csv")
for column in data.columns:
    values = data.loc[
        ~pd.isnull(data[column])
    ][column].values

    answer = values[0].strip()
    for question in values[1:]:
        questions.append(question)
        answers.append(answer)

model_path = "scripts/models/model-2.pickle"

classifier.train(questions, answers)
classifier.save(model_path)

# classifier.load(model_path)

questions = [
    "какая у вас зарплата?",
    "можно ли обучаться в компании?",
    "работать по выходным можно?"
]

for question in questions:
    answer = classifier.predict(question)
    print("{} : {}".format(question, answer))

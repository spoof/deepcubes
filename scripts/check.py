from intentclf.models import Embedder
from intentclf.models import IntentClassifier

import json
import os
import argparse

import pandas as pd

MODEL_STORAGE = 'scripts/models/'

embedder = Embedder(os.environ['INTENT_CLASSIFIER_MODEL'])
classifier = IntentClassifier(embedder)


def main(csv_path=None, json_path=None, trash_questions_path=None):
    new_model_id = None
    if json_path:
        questions, answers = [], []

        # parse data from json
        with open(json_path, "r") as handle:
            data = json.load(handle)

        for label, category in enumerate(data):
            answer = category["answers"][0]

            for question in category["questions"]:
                questions.append(question)
                answers.append(answer)

        classifier.train(questions, answers)
        classifier.threshold_calc(trash_questions_path)
        new_model_id = classifier.save(MODEL_STORAGE)

    if csv_path:
        questions, answers = [], []

        # parse from pandas data frame
        data = pd.read_csv(csv_path)
        for column in data.columns:
            values = data.loc[
                ~pd.isnull(data[column])
            ][column].values

            answer = values[0].strip()
            for question in values[1:]:
                questions.append(question)
                answers.append(answer)

        classifier.train(questions, answers)
        classifier.threshold_calc(trash_questions_path)
        new_model_id = classifier.save(MODEL_STORAGE)

    if new_model_id is not None:
        print('Created model with id {}'.format(new_model_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train new model of intent classifier'
    )
    parser.add_argument(
        '-j', '--json_path', required=False, type=str
    )
    parser.add_argument(
        '-c', '--csv_path', required=False, type=str
    )
    parser.add_argument(
        '-t', '--trash_questions_path', required=False, type=str
    )
    args = parser.parse_args()
    main(
        json_path=args.json_path,
        csv_path=args.csv_path,
        trash_questions_path=args.trash_questions_path
    )

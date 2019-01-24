from deepcubes.cubes import Embedder, NetworkEmbedder
from deepcubes.models import LogisticIntentClassifier

import json
import os
import argparse
import configparser

import pandas as pd

if 'SERVICE_CONF' in os.environ:
    config_file_path = os.environ['SERVICE_CONF']
else:
    print('Config file not found. Text config is used...')
    config_file_path = 'tests/data/test.conf'

config_parser = configparser.RawConfigParser()
config_parser.read(config_file_path)

MODEL_STORAGE = json.loads(
    config_parser.get('classifier-service', 'MODEL_STORAGE')
)

EMB_PATH = json.loads(config_parser.get('classifier-service', 'EMB_PATH'))

if 'http' in EMB_PATH:
    emb_type = 'NetworkEmbedder'
else:
    emb_type = 'Embedder'

LANG_TO_EMB_PATH = {lang: json.loads(path)
                    for lang, path in config_parser['embedder'].items()}

LANG_TO_TOK_MODE = {lang: json.loads(mode)
                    for lang, mode in config_parser['tokenizer'].items()}


def get_new_model_id(path):
    models_ids = [int(file_name) for file_name in os.listdir(path) if not (
        os.path.isfile(os.path.join(path, file_name))
    )]

    sorted_ids = sorted(models_ids)
    new_model_id = sorted_ids[-1] + 1 if len(sorted_ids) else 0

    return new_model_id


def main(csv_path, lang):
    questions, answers = [], []

    if csv_path:
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

    if emb_type == 'NetworkEmbedder':
        embedder = NetworkEmbedder(EMB_PATH, lang)
    else:
        embedder = Embedder(EMB_PATH)

    classifier = LogisticIntentClassifier(embedder)
    tokenizer_mode = LANG_TO_TOK_MODE[lang]
    classifier.train(answers, questions, tokenizer_mode)

    new_model_id = get_new_model_id(MODEL_STORAGE)

    classifier.save(os.path.join(MODEL_STORAGE, str(new_model_id)))

    if new_model_id is not None:
        print('Created model with id {}'.format(new_model_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train new model of intent classifier'
    )
    parser.add_argument(
        '-c', '--csv_path', required=False, type=str
    )
    parser.add_argument(
        '-l', '--lang', required=False, type=str
    )
    args = parser.parse_args()
    main(args.csv_path, args.lang)

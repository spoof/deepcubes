import argparse
import configparser
import os
from pprint import pprint

import pandas as pd

from deepcubes.models import VeraLiveDialog

from .embedders import EmbedderFactory
from .utils import get_new_model_id

if 'SERVICE_CONF' in os.environ:
    config_file_path = os.environ['SERVICE_CONF']
else:
    print('Config file not found. Text config is used...')
    config_file_path = (
        'tests/data/vera_live_dialog/vera_live_dialog.conf'
    )

config_parser = configparser.RawConfigParser()
config_parser.read(config_file_path)

MODEL_STORAGE = config_parser.get('live-dialog-service', 'MODEL_STORAGE')
EMBEDDER_PATH = config_parser.get('live-dialog-service', 'EMBEDDER_PATH')

embedder_factory = EmbedderFactory(EMBEDDER_PATH)

LANG_TO_EMB_MODE = dict(config_parser['embedder'])
LANG_TO_TOK_MODE = dict(config_parser['tokenizer'])

GENERIC_DATA_PATH = config_parser.get('live-dialog-service',
                                      'GENERIC_DATA_PATH')


def main(csv_path, lang):
    labels_settings = list()

    if csv_path:
        # parse from pandas data frame
        data = pd.read_csv(csv_path)
        for column in data.columns:
            values = data.loc[
                ~pd.isnull(data[column])
            ][column].values

            answer = values[0].strip()
            intent_phrases = list()
            for question in values[1:]:
                intent_phrases.append(question)
            labels_settings.append({
                'label': answer,
                'intent_phrases': intent_phrases,
            })

    for generic in ['yes', 'no', 'repeat']:
        labels_settings.append({
            'label': generic,
            'generics': [generic],
        })

    config = {
        'labels_settings': labels_settings,
        'not_understand_label': 'not_understand_label',
    }
    config['tokenizer_mode'] = LANG_TO_TOK_MODE[lang]
    config['embedder_mode'] = LANG_TO_EMB_MODE[lang]
    pprint(config)

    embedder = embedder_factory.create(LANG_TO_EMB_MODE[lang])

    live_dialog_model = VeraLiveDialog(embedder, GENERIC_DATA_PATH)
    live_dialog_model.train(config)

    new_model_id = get_new_model_id(MODEL_STORAGE)

    clf_params = live_dialog_model.save()
    clf_path = os.path.join(MODEL_STORAGE, '{}.cube'.format(new_model_id))

    os.makedirs(MODEL_STORAGE, exist_ok=True)
    with open(clf_path, 'w') as out:
        out.write(json.dumps(clf_params))

    if new_model_id is not None:
        print('Created live dialog model with id {}'.format(new_model_id))


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

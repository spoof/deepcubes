from flask import Flask, request, jsonify
import os
import sys
import json
import time
import logging
import argparse
import configparser

from deepcubes.embedders import EmbedderFactory
from deepcubes.models import VeraLiveDialog
from deepcubes.utils.functions import get_new_model_id

logger = logging.getLogger("LiveDialogService")
logger.setLevel(logging.INFO)

# create the logging file handler
handler = logging.FileHandler("scripts/logs/live_dialog_service.log")
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Started Live Dialog Server...")

# TODO: move to script argument
if 'SERVICE_CONF' in os.environ:
    config_file_path = os.environ['SERVICE_CONF']
else:
    logger.warning('Config file not found. Test config is used...')
    config_file_path = "tests/data/vera_live_dialog/vera_live_dialog.conf"

config_parser = configparser.ConfigParser()
config_parser.read(config_file_path)

MODEL_STORAGE = config_parser.get('live-dialog-service', 'MODEL_STORAGE')
logger.info("Model storage: {} ...".format(MODEL_STORAGE))

GENERIC_DATA_PATH = config_parser.get('live-dialog-service',
                                      'GENERIC_DATA_PATH')
logger.info("Generic data path: {} ...".format(GENERIC_DATA_PATH))

EMBEDDER_PATH = config_parser.get('live-dialog-service', 'EMBEDDER_PATH')
embedder_factory = EmbedderFactory(EMBEDDER_PATH)


models = dict()

LANG_TO_EMB_MODE = dict(config_parser['embedder'])
LANG_TO_TOK_MODE = dict(config_parser['tokenizer'])

logger.info("Prepare Flask app...")
app = Flask(__name__)


def load_model(model_id):
    logger.info("Loading intent model {} ...".format(model_id))
    model_path = os.path.join(
        MODEL_STORAGE, "{}.cube".format(model_id)
    )

    if not os.path.isfile(model_path):
        logger.error("Model {} not found".format(model_id))
        return None

    with open(model_path, 'r') as data:
        model_params = json.loads(data.read())

    model = VeraLiveDialog.load(model_params, embedder_factory)
    return model


@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if (
            request.method not in ["GET", "POST"]
            or ("model_id" not in request.args
                and "model_id" not in request.form)
            or ("query" not in request.args and "query" not in request.form)
        ):
            logger.error("Received invalid `predict` request")
            return jsonify({
                "message": ("Please sent GET or POST query"
                            "with `model_id` and `query` keys"),
            })

        logger.info("Received {} `predict` request from {}".format(
            request.method, request.remote_addr
        ))

        # parse data from json
        if 'model_id' in request.args:
            model_id = int(request.args['model_id'])
        elif 'model_id' in request.form:
            model_id = int(request.form['model_id'])
        else:
            logger.error("Received invalid json object")
            return jsonify({"message": "Please send correct json object"})

        logger.info("Received model id: {}".format(model_id))
        if model_id in models:
            model = models[model_id]
        else:
            model = load_model(model_id)

            if model is None:
                return jsonify({
                    "message": "Model with model_id {} not found".format(
                        model_id),
                })
            else:
                models[model_id] = model

        if 'query' in request.args:
            query = request.args['query']
        elif 'query' in request.form:
            query = request.form['query']
        else:
            logger.error("Received invalid json object")
            return jsonify({"message": "Please send correct json object"})

        if 'labels' in request.args:
            labels = request.args['labels']
        elif 'labels' in request.form:
            labels = request.form['labels']
        else:
            labels = list()

        logger.info('Received query: {}'.format(query))
        logger.info('Received labels: {}'.format(labels))

        model_answer = model(query, labels)

        output = []
        for label, probability in model_answer:
            output.append({
                "label": label,
                "proba": probability,
            })

        logger.info("Top predicted label: {}".format(output[0]['label']))
        logger.info("Max probability: {}".format(output[0]['proba']))

        with open('scripts/logs/live_dialog_service.json', 'a') as out:
            data = {
                'timestamp': time.time(),
                'model_path': os.path.join(
                    MODEL_STORAGE, '{}.cube'.format(model_id)
                ),
                'query': query,
                'labels': labels,
                'answer': output,
            }

            print(json.dumps(data), file=out)

        logger.info("Sending response...")

        return jsonify(output)

    except Exception:
        _, exc_obj, exc_tb = sys.exc_info()
        logger.error("line {}, {}".format(exc_tb.tb_lineno, exc_obj))


@app.route("/train", methods=["GET", "POST"])
def train():
    try:
        if (
            request.method not in ["GET", "POST"]
            or ("config" not in request.args and "config" not in request.form)
        ):
            logger.error("Received invalid `train` request")
            return jsonify({
                "message": ("Please sent GET or POST query with `config` key"),
            })

        logger.info("Received {} `train` request from {}".format(
            request.method, request.remote_addr
        ))

        # parse data from json
        if 'config' in request.args:
            config_string = request.args["config"]
        elif 'config' in request.form:
            config_string = request.form["config"]
        else:
            logger.error("Received invalid json object")
            return jsonify({"message": "Please send correct json object"})

        try:
            config = json.loads(config_string)
        except Exception as e:
            print(repr(e))
            return jsonify({"message": "Config in wrong format."})

        logger.info("Received `lang` key: {}".format(config['lang']))

        embedder_mode = LANG_TO_EMB_MODE[config['lang']]
        tokenizer_mode = LANG_TO_TOK_MODE[config['lang']]

        embedder = embedder_factory.create(embedder_mode)
        logger.info("Set embedder mode: {}".format(embedder_mode))

        config['embedder_mode'] = embedder_mode
        config['tokenizer_mode'] = LANG_TO_TOK_MODE[config['lang']]

        logger.info("`{}` tokenizer mode set".format(tokenizer_mode))

        live_dialog_model = VeraLiveDialog(embedder, GENERIC_DATA_PATH)
        live_dialog_model.train(config)

        new_model_id = get_new_model_id(MODEL_STORAGE)

        clf_params = live_dialog_model.save()
        clf_path = os.path.join(MODEL_STORAGE, '{}.cube'.format(new_model_id))

        os.makedirs(MODEL_STORAGE, exist_ok=True)
        with open(clf_path, 'w') as out:
            out.write(json.dumps(clf_params))

        models[new_model_id] = live_dialog_model

        logging.info('Saved model with model_id {}'.format(new_model_id))

        return jsonify({
            "message": 'Created model with model_id {}'.format(new_model_id),
            "model_id": new_model_id,
        })

    except Exception:
        _, exc_obj, exc_tb = sys.exc_info()
        logger.error("line {}, {}".format(exc_tb.tb_lineno, exc_obj))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Launch Vera Live Dialog API'
    )

    parser.add_argument(
        '-m', '--model_id_list', nargs='+', type=int, default=list()
    )

    args = parser.parse_args()
    for model_id in args.model_id_list:
        model = load_model(model_id)

        if model is None:
            print("Model with id {} not found".format(model_id))
        else:
            models[model_id] = model

    app.run(host="0.0.0.0", port=3335, debug=False)

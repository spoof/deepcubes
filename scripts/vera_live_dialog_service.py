from flask import Flask, request, jsonify
import os
import sys
import json
import time
import logging
import argparse
import configparser

from deepcubes.embedders import Embedder, NetworkEmbedder
from deepcubes.models import VeraLiveDialog

logger = logging.getLogger("LiveDialogService")
logger.setLevel(logging.INFO)

# create the logging file handler
fh = logging.FileHandler("scripts/logs/live_dialog_service.log")
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s'
)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("Started Live Dialog Server...")

if 'SERVICE_CONF' in os.environ:
    config_file_path = os.environ['SERVICE_CONF']
else:
    logger.warning('Config file not found. Test config is used...')
    config_file_path = 'tests/data/test.conf'

config_parser = configparser.RawConfigParser()
config_parser.read(config_file_path)

MODEL_STORAGE = json.loads(
    config_parser.get('live-dialog-service', 'MODEL_STORAGE')
)
logger.info("Model storage: {} ...".format(MODEL_STORAGE))

EMB_PATH = json.loads(config_parser.get('live-dialog-service', 'EMB_PATH'))
logger.info("Embedder path: {} ...".format(MODEL_STORAGE))

if 'http' in EMB_PATH:
    emb_type = 'NetworkEmbedder'
else:
    emb_type = 'Embedder'

GENERIC_DATA_PATH = json.loads(
    config_parser.get('live-dialog-service', 'GENERIC_DATA_PATH')
)
logger.info("Generic data path: {} ...".format(GENERIC_DATA_PATH))

models = dict()

LANG_TO_EMB_PATH = {lang: json.loads(path)
                    for lang, path in config_parser['embedder'].items()}

LANG_TO_TOK_MODE = {lang: json.loads(mode)
                    for lang, mode in config_parser['tokenizer'].items()}

logger.info("Prepare Flask app...")
app = Flask(__name__)


def load_model(model_id):
    logger.info("Loading intent model {} ...".format(model_id))
    model_path = os.path.join(
        MODEL_STORAGE, "{}/vera_live_dialog.cube".format(model_id)
    )

    if not os.path.isfile(model_path):
        logger.error(
            "Model {} not found".format(model_id)
        )
        return False

    if model_id not in models:
        model = VeraLiveDialog.load(model_path)
        models[model_id] = model

    return True


def get_new_model_id(path):
    models_ids = [int(file_name) for file_name in os.listdir(path) if not (
        os.path.isfile(os.path.join(path, file_name))
    )]

    sorted_ids = sorted(models_ids)
    new_model_id = sorted_ids[-1] + 1 if len(sorted_ids) else 0

    return new_model_id


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
            loading = load_model(model_id)

            if loading:
                model = models[model_id]
            else:
                return jsonify({
                    "message": "Model with model_id {} not found".format(
                        model_id),
                })

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
                'model_path': model.cube_path,
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
        embedder_mode = config['lang']
        tokenizer_mode = LANG_TO_TOK_MODE[config['lang']]

        if emb_type == 'NetworkEmbedder':
            embedder = NetworkEmbedder(EMB_PATH, embedder_mode)
            logger.info("Set {}".format(emb_type))
        else:
            embedder = Embedder(EMB_PATH)
            logger.info("Set {} with mode `{}`".format(emb_type,
                                                       embedder_mode))

        config['embedder_mode'] = embedder_mode
        config['tokenizer_mode'] = LANG_TO_TOK_MODE[config['lang']]

        logger.info("`{}` tokenizer mode set".format(tokenizer_mode))

        live_dialog_model = VeraLiveDialog(embedder, GENERIC_DATA_PATH)
        live_dialog_model.train(config)

        new_model_id = get_new_model_id(MODEL_STORAGE)
        models[new_model_id] = live_dialog_model

        model_path = os.path.join(MODEL_STORAGE, str(new_model_id))
        live_dialog_model.save(model_path)

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
        loading = load_model(model_id)

        if not loading:
            print("Model with id {} not found".format(model_id))

    app.run(host="0.0.0.0", port=3335, debug=False)

from flask import Flask, request, jsonify
import os
import sys
import json
import time
import logging
import argparse
import configparser

from deepcubes.models import LogisticIntentClassifier
from deepcubes.embedders import EmbedderFactory

logger = logging.getLogger("ClassifierService")
logger.setLevel(logging.INFO)

# create the logging file handler
handler = logging.FileHandler("scripts/logs/classifier_service.log")
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Started Intent Classifier Server...")

if 'SERVICE_CONF' in os.environ:
    config_file_path = os.environ['SERVICE_CONF']
else:
    logger.warning('Config file not found. Test config is used...')
    config_file_path = 'tests/data/classifier_service/classifier_service.conf'

logger.info("Read config file {} ...".format(config_file_path))
config_parser = configparser.RawConfigParser()
config_parser.read(config_file_path)

MODEL_STORAGE = config_parser.get('classifier-service', 'MODEL_STORAGE')

logger.info("Model storage: {} ...".format(MODEL_STORAGE))

EMBEDDER_PATH = config_parser.get('classifier-service', 'EMBEDDER_PATH')

if 'http' in EMBEDDER_PATH:
    embedder_factory = EmbedderFactory(network_url=EMBEDDER_PATH)
else:
    embedder_factory = EmbedderFactory(local_path=EMBEDDER_PATH)


models = dict()

logger.info("Prepare Flask app...")
app = Flask(__name__)


def load_model(model_id):
    logger.info("Loading intent model {} ...".format(model_id))
    model_path = os.path.join(
        MODEL_STORAGE, "{}/intent_classifier.cube".format(model_id)
    )

    if not os.path.isfile(model_path):
        logger.error("Model {} not found".format(model_id))
        return None

    model = LogisticIntentClassifier.load(model_path, embedder_factory)

    return model


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
            logger.error("Received invalid request")
            return jsonify({
                "message": ("Please sent GET or POST query"
                            "with `model_id` and `query` keys"),
            })

        logger.info("Received {} request from {}".format(request.method,
                                                         request.remote_addr))

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

            if model is not None:
                models[model_id] = model
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

        logger.info('Received query: {}'.format(query))

        model_answer = model(query)

        output = []
        for label, probability in model_answer:
            output.append({
                "answer": label,
                "probability": probability,
                "threshold": 0.3,
                "accuracy_score": None
            })

        logger.info("Top predicted label: {}".format(output[0]['answer']))
        logger.info("Max probability: {}".format(output[0]['probability']))

        with open('scripts/logs/classifier_service.json', 'a') as out:
            data = {
                'timestamp': time.time(),
                'model_path': model.cube_path,
                'query': query,
                'answer': output,
            }

            print(json.dumps(data), file=out)

        logger.info("Sending response...")
        return jsonify(output)

    except Exception:
        _, exc_obj, exc_tb = sys.exc_info()
        logger.error("line {}, {}".format(exc_tb.tb_lineno, exc_obj))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Launch Vera Live Dialog API'
    )

    parser.add_argument('-m', '--model_id_list',
                        nargs='+', type=int, default=list())

    args = parser.parse_args()

    for model_id in args.model_id_list:
        model = load_model(model_id)

        if model is not None:
            models[model_id] = model
        else:
            print("Error while loading {} model_id".format(model_id))

    app.run(host="0.0.0.0", port=3337, debug=False)
